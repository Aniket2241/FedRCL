import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from collections import defaultdict
import copy
import gc
import time

# Import the base Client class
from clients.base_client import Client  # Update this path as necessary to reflect your project structure
from clients.build import CLIENT_REGISTRY  # Ensure you import your client registry

@CLIENT_REGISTRY.register()
class RCLClient(Client):

    def __init__(self, args, client_index, model):
        super().__init__(args, client_index, model)
        self.residuals = {}  # Residuals for sparsification

    def sparsify(self, updates, k):
        """Sparsify the updates by retaining only top-k elements."""
        flat_updates = torch.cat([param.view(-1) for param in updates.values()])
        topk_values, topk_indices = torch.topk(torch.abs(flat_updates), k)
        mask = torch.zeros_like(flat_updates)
        mask[topk_indices] = 1

        sparsified_updates = {}
        start = 0
        for name, param in updates.items():
            numel = param.numel()
            sparsified_updates[name] = (flat_updates[start:start+numel].view_as(param) *
                                        mask[start:start+numel].view_as(param))
            start += numel
        
        return sparsified_updates

    def local_train(self, global_epoch, **kwargs):
        self.global_epoch = global_epoch

        self.model.to(self.device)
        self.global_model.to(self.device)

        scaler = GradScaler()
        start = time.time()

        self.weights = self.get_weights(epoch=global_epoch)
        
        for local_epoch in range(self.args.trainer.local_epochs):
            for i, (images, labels) in enumerate(self.loader):
                images, labels = images.to(self.device), labels.to(self.device)
                self.model.zero_grad()

                with autocast(enabled=self.args.use_amp):
                    losses, _ = self._algorithm(images, labels)

                    loss = sum([self.weights.get(loss_key, 0) * losses[loss_key] for loss_key in losses])

                scaler.scale(loss).backward()
                scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
                scaler.step(self.optimizer)
                scaler.update()

            self.scheduler.step()

        # Compute the model updates
        local_updates = {
            name: param - self.global_model.state_dict()[name]
            for name, param in self.model.state_dict().items()
        }

        # Add residuals to current updates
        for name in local_updates:
            if name in self.residuals:
                local_updates[name] += self.residuals[name]

        # Sparsify the updates
        sparsified_updates = self.sparsify(local_updates, k=self.args.client.top_k)

        # Update residuals
        self.residuals = {
            name: local_updates[name] - sparsified_updates[name]
            for name in local_updates
        }

        self.model.to('cpu')
        self.global_model.to('cpu')

        gc.collect()

        return sparsified_updates, {}
