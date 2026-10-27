"""Check the behavior of the locally patched dependency on small CPU inputs."""
import copy
import unittest
from unittest.mock import patch

import torch
from transformers import BertConfig, BertModel

from groundingdino.models.GroundingDINO.backbone.swin_transformer import BasicLayer
from groundingdino.models.GroundingDINO.bertwarper import BertModelWarper


class GroundingDINOCompatibilityTests(unittest.TestCase):
    def make_layer(self):
        torch.manual_seed(0)
        return BasicLayer(dim=16, depth=2, num_heads=2, window_size=2, use_checkpoint=True)

    def test_eval_does_not_checkpoint(self):
        layer = self.make_layer().eval()
        with torch.no_grad(), patch("torch.utils.checkpoint.checkpoint", side_effect=AssertionError("checkpoint in inference")):
            result = layer(torch.randn(1, 16, 16), 4, 4)[0]
        self.assertEqual(tuple(result.shape), (1, 16, 16))
        self.assertTrue(torch.isfinite(result).all())

    def test_checkpoint_preserves_training_outputs_and_gradients(self):
        layer = self.make_layer().train()
        reference = copy.deepcopy(layer)
        reference.use_checkpoint = False
        x = torch.randn(1, 16, 16, requires_grad=True)
        x_reference = x.detach().clone().requires_grad_(True)
        actual = layer(x, 4, 4)[0]
        expected = reference(x_reference, 4, 4)[0]
        torch.testing.assert_close(actual, expected)
        actual.square().mean().backward()
        expected.square().mean().backward()
        torch.testing.assert_close(x.grad, x_reference.grad)

    def test_bert_attention_mask_preserves_encoder_output(self):
        config = BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
                            num_attention_heads=2, intermediate_size=32)
        original = BertModel(config).eval()
        wrapped = BertModelWarper(original).eval()
        ids = torch.tensor([[1, 2, 3, 0]])
        mask = torch.tensor([[1, 1, 1, 0]])
        with torch.no_grad():
            actual = wrapped(input_ids=ids, attention_mask=mask).last_hidden_state
            expected = original(input_ids=ids, attention_mask=mask).last_hidden_state
        torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    unittest.main()
