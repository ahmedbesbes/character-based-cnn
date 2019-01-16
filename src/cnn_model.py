import json
import torch.nn as nn


class CharacterLevelCNN(nn.Module):
    def __init__(self, config_path):
        super(CharacterLevelCNN, self).__init__()

        with open(config_path) as f:
            self.config = json.load(f)

        conv_layers = []
        for i, conv_layer_parameter in enumerate(self.config['model_parameters']['conv']):
            if i == 0:
                in_channels = len(self.config['alphabet'])
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

        # dimension = int((self.config['max_length'] - 96) / 27 *
                        # self.config['model_parameters']['conv'][-1][0])

        dimension = 256
        
        fc_layers = []
        for i, fc_layer_parameter in enumerate(self.config['model_parameters']['fc']):
            if i == 0:
                fc_layer = nn.Sequential(
                    nn.Linear(dimension, fc_layer_parameter), nn.Dropout(0.5))
            elif i == len(self.config['model_parameters']['fc']) - 1:
                fc_layer = nn.Sequential(
                    nn.Linear(fc_layer_parameter, self.config['num_of_classes']))
            else:
                fc_layer = nn.Sequential(
                    nn.Linear(fc_layer_parameter, fc_layer_parameter), nn.Dropout(0.5))
            fc_layers.append(fc_layer)
        self.fc_layers = nn.ModuleList(fc_layers)

        if self.config['model_parameters']['name'] == 'small':
            self._create_weights(mean=0.0, std=0.05)
        elif self.config['model_parameters']['name'] == 'large':
            self._create_weights(mean=0.0, std=0.02)

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

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
