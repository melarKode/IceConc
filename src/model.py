import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, bias):
        """
        input_size: channels of input tensor
        hidden_size: channels of hidden state
        kernel_size: size of the convolutional kernel
        bias: whether to use bias in the convolutional layer
        """
        super(ConvLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.bias = bias
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2

        self.conv = nn.Conv2d(in_channels=self.input_size + self.hidden_size,
                              out_channels=4 * self.hidden_size,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

        self.Wci = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=1, bias=self.bias)
        self.Wcf = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=1, bias=self.bias)
        self.Wco = nn.Conv2d(in_channels=self.hidden_size, out_channels=self.hidden_size, kernel_size=1, bias=self.bias)

        
    def forward(self, x, curr_state):
        h_cur, c_cur = curr_state
        combined = torch.cat([x, h_cur], dim=1)
        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv,
                                             split_size_or_sections=self.hidden_size,
                                             dim=1)
        i = torch.sigmoid(cc_i + self.Wci(c_cur))
        f = torch.sigmoid(cc_f + self.Wcf(c_cur))
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        
        o = torch.sigmoid(cc_o + self.Wco(c_next))

        h_next = o * torch.tanh(c_next)

        return h_next, c_next

class ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, num_layers, bias=True, return_sequences=False):
        super(ConvLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = self._multilayer_support(hidden_size, num_layers)
        self.kernel_size = self._multilayer_support(kernel_size, num_layers)
        self.num_layers = num_layers
        self.bias = bias
        self.return_sequences = return_sequences
        cell_list = []
        
        for i in range(0, self.num_layers):
            curr_input_size = self.input_size if i == 0 else self.hidden_size[i-1]
            cell_list.append(ConvLSTMCell(input_size=curr_input_size,
                                          hidden_size=self.hidden_size[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))
            
        self.cell_list = nn.ModuleList(cell_list)

        self.output_projection = nn.Conv2d(in_channels=self.hidden_size[-1],
                                           out_channels=self.input_size,
                                           kernel_size=1,
                                           bias=self.bias)
        
        self.output_activation = nn.Sigmoid()

    def forward(self, x, hidden_state=None):
        b, seq_len, _, h, w = x.size()

        if hidden_state is None:
            hidden_state = self._init_hidden(b, (h, w), x.device)
        else:
            pass

        last_layer_output_sequence = []

        for t in range(seq_len):
            x_t = x[:,t,:,:,:]

            next_hidden_t = []

            #unrolling

            for idx in range(self.num_layers):
                cell = self.cell_list[idx]

                h_cur, c_cur = hidden_state[idx]
                h_next, c_next = cell(x=x_t, curr_state=[h_cur, c_cur])

                next_hidden_t.append((h_next, c_next))
                
                x_t = h_next
            
            hidden_state = next_hidden_t
            last_layer_output_sequence.append(x_t)

        last_layer_output_sequence = torch.stack(last_layer_output_sequence, dim = 0)
        #now seq_len is the first dimension
        last_layer_output_sequence = last_layer_output_sequence.permute(1, 0, 2, 3, 4)

        if self.return_sequences:
            projected_output = self.output_projection(last_layer_output_sequence.view(-1, self.hidden_size[-1], h, w))
            projected_output = projected_output.view(b, seq_len, self.input_size, h, w)
        else:
            projected_output = self.output_projection(last_layer_output_sequence[:,-1,:,:,:])
        
        
        final_prediction = self.output_activation(projected_output)

        return final_prediction


    def _init_hidden(self, batch_size, image_size, device):
        H, W = image_size
        init_states = []
        for i in range(self.num_layers):
            init_states.append((torch.zeros(batch_size, self.hidden_size[i], H, W, device=device),
                                torch.zeros(batch_size, self.hidden_size[i], H, W, device=device)))
        return init_states

    @staticmethod
    def _multilayer_support(x, num_layers):
        if (not isinstance(x, list)):
            x = [x] * num_layers
        return x
        