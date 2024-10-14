import torch
import torch.nn as nn


class LSTMAutoEncoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, batch_size, dropout=0, device=torch.device('cpu')):
        super(LSTMAutoEncoder, self).__init__()
        self.device = device
        #self.encoder = Encoder(num_layers, hidden_size, nb_feature, batch_size, dropout=dropout, device=device)    #todo: CHANGE!!!!!!!!!!!!!!!!!!!
        self.encoder = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)
        self.decoder = Decoder(num_layers, hidden_size, nb_feature, dropout=dropout, device=device)
        self.is_training = None

    def forward(self, input_seq, step_window):

        output = torch.ones(size=input_seq.shape, dtype=torch.float)
        _, last_hidden = self.encoder(input_seq)
        input_decoder = input_seq[:, -1, :].view(input_seq.shape[0], 1, input_seq.shape[2])
        output = torch.ones(size=(input_seq.shape[0], step_window, input_seq.shape[2]), dtype=torch.float)

        for i in range(step_window - 1, -1, -1):
            output_decoder, last_hidden = self.decoder(input_decoder, last_hidden)  # hidden_cell[0] doesnt work
            input_decoder = output_decoder
            output[:, i, :] = output_decoder[:, 0, :]

        #print(f"output_decoder_shape: {output_decoder.cpu().detach().numpy().shape}")
        #print(f"output var in loop: {output_decoder.cpu().detach().numpy().shape}")


        #return output
        #print(f"output_decoder_shape: {output_decoder.cpu().detach().numpy().shape}")
        #print(f"output[:, 0, :]_shape :,0,: : {output[:, 0, :].cpu().detach().numpy().shape}")
        #print(f"output_shape: {output.cpu().detach().numpy().shape}")

        return output


class Decoder(nn.Module):
    def __init__(self, num_layers, hidden_size, nb_feature, dropout=0, device=torch.device('cpu')):
        super(Decoder, self).__init__()

        self.input_size = nb_feature
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device

        self.lstm = nn.LSTM(input_size=nb_feature, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, dropout=dropout, bias=True)

        self.linear = nn.Linear(in_features=hidden_size, out_features=nb_feature)

    def forward(self, input_seq, hidden_cell):
        output, hidden_cell = self.lstm(input_seq, hidden_cell)
        output = self.linear(output)

        return output, hidden_cell




