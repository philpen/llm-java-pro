
import logging
import array
import struct
import numpy as np

log = logging.getLogger(__name__)


class Config:
    headers = 256
    header_size_b = headers * 4
    vocab_size = 0
    num_layers = 0
    num_heads = 0
    channels = 0


class ParameterTensors:
    def __init__(self, file_name, config):

        f = open(file_name, 'rb')
        f.seek(1024,0)

        self.file_size = f.tell()
        assert((self.file_size - 1024) % 4 == 0)
        self.num_params = int((self.file_size - 1024)/4)
        self.mem = np.fromfile(f, np.float32)
        struct.unpack('f' * self.num_params, f.read(4 * self.num_params))
        self.maxT = config.max_seq_len
        self.C = config.channels
        self.V = config.vocab_size
        self.L = config.num_layers

        self.wte_size = self.V * self.C
        self.wte = 0

        self.wpe_size = self.maxT * self.C
        self.wpe = self.wte + self.wte_size

        self.ln1w_size = self.L * self.C
        self.ln1w = self.wpe + self.wpe_size

        self.ln1b_size = self.L * self.C
        self.ln1b = self.ln1w + self.ln1w_size

        self.qkvw_size = self.L * (3 * self.C) * self.C
        self.qkvw = self.ln1b + self.ln1b_size

        self.qkvb_size = self.L * (3 * self.C)
        self.qkvb = self.qkvw + self.qkvw_size

        self.attprojw_size = self.L * self.C * self.C
        self.attprojw = self.qkvb + self.qkvb_size

        self.attprojb_size = self.L * self.C
        self.attprojb = self.attprojw + self.attprojw_size

        self.ln2w_size = self.L * self.C
        self.ln2w = self.attprojb + self.attprojb_size

        self.ln2b_size = self.L * self.C
        self.ln2b = self.ln2w + self.ln2w_size

        self.fcw_size = self.L * (4 * self.C) * self.C
        self.fcw = self.ln2b + self.ln2b_size

        self.fcb_size = self.L * (4 * self.C)
        self.fcb = self.fcw + self.fcw_size

        self.fcprojw_size = self.L * self.C * (4 * self.C)
        self.fcprojw = self.fcb + self.fcb_size

        self.fcprojb_size = self.L * self.C
        self.fcprojb = self.fcprojw + self.fcprojw_size

        self.lnfw_size = self.C
        self.lnfw = self.fcprojb + self.fcprojb_size

        self.lnfb_size = self.C
        self.lnfb = self.lnfw + self.lnfw_size

        self.num_params = self.lnfb + self.lnfb_size
        log.info(f"num_params {self.num_params}")
        self.run_assertions()

    def getMem(self, ix):
        return self.mem[ix]

    def run_assertions(self):
        v = self.mem[self.wte]
        print(f'wte[0] == {v}')
        assert(self.mem[self.wte] == -0.11010301113128662)
        assert(self.mem[self.wte + 5] == -0.078917674720287323)

        assert (self.mem[self.wpe] == -0.018820719793438911)