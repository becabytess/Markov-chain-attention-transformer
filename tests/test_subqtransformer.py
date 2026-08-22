"""
Comprehensive Unit & Integration Tests for SubQTransformer.
"""

import unittest
import torch
from subqtransformer import (
    SubQConfig,
    SubQSurfer,
    SubQBlock,
    SubQTransformerLM,
    SubQTransformerClassifier
)
import gravimem


class TestSubQTransformer(unittest.TestCase):
    def setUp(self):
        self.config = SubQConfig(
            vocab_size=256,
            d_model=64,
            n_heads=4,
            n_layers=2,
            default_T=3,
            max_seq_len=128,
            d_mlp=128,
            jump_offsets=[0, 1, 2, 4, 8, 16, 32, 64]
        )

    def test_config_initialization(self):
        cfg = SubQConfig(vocab_size=1000, d_model=128)
        self.assertEqual(cfg.d_mlp, 512)
        self.assertTrue(len(cfg.jump_offsets) > 0)

    def test_lm_forward_and_loss(self):
        model = SubQTransformerLM(self.config)
        x = torch.randint(0, self.config.vocab_size, (2, 32))
        y = torch.randint(0, self.config.vocab_size, (2, 32))

        logits, loss = model(x, targets=y)
        self.assertEqual(logits.shape, (2, 32, self.config.vocab_size))
        self.assertIsNotNone(loss)
        self.assertFalse(torch.isnan(loss))

        # Test backward pass
        loss.backward()
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad, f"Grad is None for {name}")

    def test_adaptive_halting_and_stats(self):
        model = SubQTransformerLM(self.config)
        x = torch.randint(0, self.config.vocab_size, (2, 16))

        logits, stats = model(x, adaptive_halting=True, halt_threshold=0.08, return_stats=True)
        self.assertEqual(logits.shape, (2, 16, self.config.vocab_size))
        self.assertIn("avg_hops_per_layer", stats)
        self.assertIn("mean_total_hops", stats)
        self.assertTrue(stats["mean_total_hops"] > 0)

    def test_generation(self):
        model = SubQTransformerLM(self.config)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)
        out = model.generate(prompt, max_new_tokens=10, temperature=0.8, top_k=5)
        self.assertEqual(out.shape, (1, 13))

    def test_classifier(self):
        model = SubQTransformerClassifier(num_classes=4, config=self.config)
        x = torch.randint(0, self.config.vocab_size, (3, 20))
        y = torch.tensor([0, 1, 3], dtype=torch.long)

        logits, loss = model(x, targets=y)
        self.assertEqual(logits.shape, (3, 4))
        self.assertIsNotNone(loss)

    def test_gravimem_backward_compatibility(self):
        lm = gravimem.GravimemLM(vocab_size=256, d_model=64, max_seq_len=128)
        x = torch.randint(0, 256, (2, 16))
        out = lm(x)
        self.assertEqual(out.shape, (2, 16, 256))


if __name__ == "__main__":
    unittest.main()
