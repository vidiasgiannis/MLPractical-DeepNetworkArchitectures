import unittest
import os
import torch
from model_architectures import (
    RC_BN_ConvolutionalProcessingBlock,
    BN_ConvolutionalProcessingBlock,
    BN_ConvolutionalDimensionalityReductionBlock
)

class TestModelArchitectures(unittest.TestCase):
    def setUp(self):
        """
        Simulate command-line arguments typically used for training the model.
        These arguments are used to configure the test cases for consistency with the training script.
        """
        self.config = {
            "batch_size": 100,
            "seed": 0,
            "num_filters": 32,
            "num_stages": 3,
            "num_blocks_per_stage": 5,
            "use_gpu": True,
            "num_classes": 100,
            "block_type": "conv_block",
        }
        self.output_dir = "block_test_results"
        os.makedirs(self.output_dir, exist_ok=True)  # Create the output folder if it doesn't exist
        torch.manual_seed(self.config["seed"])

    def save_results(self, block_name, x, output):
        """
        Save the input and output shapes and example values to a file.
        """
        filepath = os.path.join(self.output_dir, f"{block_name}_results.txt")
        with open(filepath, "w") as f:
            f.write(f"{block_name}\n")
            f.write(f"Input Shape: {x.shape}\n")
            f.write(f"Output Shape: {output.shape}\n")
            f.write(f"Example Input Values:\n{x[0, 0, :3, :3]}\n")
            f.write(f"Example Output Values:\n{output[0, 0, :3, :3]}\n")

    def test_rc_bn_convolutional_processing_block(self):
        input_shape = (self.config["batch_size"], 32, 32, 32)  # Match num_filters
        num_filters = 32  # Match input channels to avoid mismatch
        kernel_size = 3
        padding = 1
        bias = True
        dilation = 1

        block = RC_BN_ConvolutionalProcessingBlock(
            input_shape=input_shape,
            num_filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation
        )

        x = torch.randn(input_shape)
        output = block(x)

        self.assertEqual(output.shape, input_shape)
        self.save_results("RC_BN_ConvolutionalProcessingBlock", x, output)
        print("\n[RC_BN_ConvolutionalProcessingBlock]")
        print(f"Input Shape: {x.shape}")
        print(f"Output Shape: {output.shape}")

    def test_bn_convolutional_processing_block(self):
        input_shape = (self.config["batch_size"], 3, 32, 32)
        num_filters = self.config["num_filters"]
        kernel_size = 3
        padding = 1
        bias = True
        dilation = 1

        block = BN_ConvolutionalProcessingBlock(
            input_shape=input_shape,
            num_filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation
        )

        x = torch.randn(input_shape)
        output = block(x)

        expected_shape = (self.config["batch_size"], num_filters, 32, 32)
        self.assertEqual(output.shape, expected_shape)
        self.save_results("BN_ConvolutionalProcessingBlock", x, output)
        print("\n[BN_ConvolutionalProcessingBlock]")
        print(f"Input Shape: {x.shape}")
        print(f"Output Shape: {output.shape}")

    def test_bn_convolutional_dimensionality_reduction_block(self):
        input_shape = (self.config["batch_size"], 3, 32, 32)
        num_filters = self.config["num_filters"]
        kernel_size = 3
        padding = 1
        bias = True
        dilation = 1
        reduction_factor = 2

        block = BN_ConvolutionalDimensionalityReductionBlock(
            input_shape=input_shape,
            num_filters=num_filters,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
            dilation=dilation,
            reduction_factor=reduction_factor
        )

        x = torch.randn(input_shape)
        output = block(x)

        expected_shape = (self.config["batch_size"], num_filters, 16, 16)
        self.assertEqual(output.shape, expected_shape)
        self.save_results("BN_ConvolutionalDimensionalityReductionBlock", x, output)
        print("\n[BN_ConvolutionalDimensionalityReductionBlock]")
        print(f"Input Shape: {x.shape}")
        print(f"Output Shape: {output.shape}")


if __name__ == "__main__":
    unittest.main()
