import json
import torch
import torch.nn as nn


class CharacterLevelCNN(nn.Module):
    def __init__(self, args):
        super(CharacterLevelCNN, self).__init__()

        with open(args.config_path) as f:
            self.config = json.load(f)

        conv_layers = []
        for i, conv_layer_parameter in enumerate(self.config['model_parameters'][args.size]['conv']):
            if i == 0:
                in_channels = args.number_of_characters
                out_channels = conv_layer_parameter[0]
            else:
                in_channels, out_channels = conv_layer_parameter[0], conv_layer_parameter[0]

            if conv_layer_parameter[2] != -1:
                conv_layer = nn.Sequential(nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=conv_layer_parameter[1], padding=0),
                                           nn.ReLU(),
                                           nn.MaxPool1d(conv_layer_parameter[2]))
            else:
                conv_layer = nn.Sequential(nn.Conv1d(in_channels,
                                                     out_channels,
                                                     kernel_size=conv_layer_parameter[1], padding=0),
                                           nn.ReLU())
            conv_layers.append(conv_layer)
        self.conv_layers = nn.ModuleList(conv_layers)

        input_shape = (args.batch_size, args.max_length,
                       args.number_of_characters)
        dimension = self._get_conv_output(input_shape)

        print('dimension :', dimension)

        fc_layer_parameter = self.config['model_parameters'][args.size]['fc'][0]
        fc_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dimension, fc_layer_parameter), nn.Dropout(0.5)),
            nn.Sequential(nn.Linear(fc_layer_parameter,
                                    fc_layer_parameter), nn.Dropout(0.5)),
            nn.Linear(fc_layer_parameter, args.number_of_classes),
        ])

        self.fc_layers = fc_layers

        if args.size == 'small':
            self._create_weights(mean=0.0, std=0.05)
        elif args.size == 'large':
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def _get_conv_output(self, shape):
        input = torch.rand(shape)
        output = input.transpose(1, 2)
        # forward pass through conv layers
        for i in range(len(self.conv_layers)):
            output = self.conv_layers[i](output)

        output = output.view(output.size(0), -1)
        n_size = output.size(1)
        return n_size

    def forward(self, input):
        output = input.transpose(1, 2)
        # forward pass through conv layers
        for i in range(len(self.conv_layers)):
            output = self.conv_layers[i](output)

        output = output.view(output.size(0), -1)

        # forward pass through fc layers
        for i in range(len(self.fc_layers)):
            output = self.fc_layers[i](output)
        return output