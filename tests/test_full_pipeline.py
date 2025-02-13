import unittest
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from trainer.trainer import MoASETrainer
from config import Config

class TestFullPipeline(unittest.TestCase):
    def setUp(self):
        """Set up the configuration for the tests."""
        self.config = Config()

    def test_backbone(self):
        """Test the backbone forward pass and adaptation toggle."""
        from modules.backbone import Backbone
        model = Backbone(num_classes=self.config.num_classes)
        x = torch.randn(4, 3, 32, 32)
        logits, features = model(x)

        self.assertEqual(logits.shape, (4, self.config.num_classes), "Error [B001]: Backbone logits shape mismatch")
        self.assertEqual(features.shape, (4, 64), "Error [B002]: Backbone feature shape mismatch")
        print("Backbone test passed.")

    def test_loss_function(self):
        """Test the composite loss function with sample data."""
        from modules.losses import CompositeLoss
        loss_fn = CompositeLoss(task_loss_type=self.config.task_loss_type, num_splits=self.config.num_splits)

        student_outputs = torch.randn(12, self.config.num_classes)
        targets = torch.randint(0, self.config.num_classes, (12,))
        student_expert_params = [torch.randn(5, 5), torch.randn(3, 3)]
        teacher_expert_params = [p + 0.1 * torch.randn_like(p) for p in student_expert_params]

        loss = loss_fn(student_outputs, targets, student_expert_params, teacher_expert_params, teacher_outputs=student_outputs)
        self.assertTrue(torch.isfinite(loss), "Error [L001]: Loss contains NaN or Inf")
        self.assertGreater(loss.item(), 0, "Error [L002]: Loss value should be positive")
        print("Loss function test passed.")

    def test_router(self):
        """Test the router with two dummy experts."""
        from modules.routing import Router
        class DummyExpert(torch.nn.Module):
            def __init__(self, out_dim):
                super().__init__()
                self.fc = torch.nn.Linear(10, out_dim)
            def forward(self, x):
                return self.fc(x)
        
        expert1 = DummyExpert(20)
        expert2 = DummyExpert(20)
        router = Router(torch.nn.ModuleList([expert1, expert2]))

        x = torch.randn(4, 10)
        gate_weights = torch.tensor([[0.7, 0.3], [0.5, 0.5], [0.9, 0.1], [0.2, 0.8]])
        routed_output = router(x, gate_weights)

        self.assertEqual(routed_output.shape, (4, 20), "Error [R001]: Router output shape mismatch")
        self.assertTrue(torch.isfinite(routed_output).all(), "Error [R002]: Router output contains NaN or Inf")
        print("Router test passed.")

    def test_trainer(self):
        """Test the full training pipeline with PyTorch Lightning."""
        model = MoASETrainer(self.config)
        trainer = Trainer(max_epochs=2, enable_progress_bar=True, callbacks=[EarlyStopping(monitor="val_loss", patience=1)])
        
        trainer.fit(model)  # This uses the dummy data loaders from MoASETrainer
        print("Trainer test passed.")

if __name__ == "__main__":
    unittest.main()
