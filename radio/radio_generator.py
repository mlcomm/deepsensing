import math
import numpy as np
import random
from gnuradio import gr, digital
from gnuradio import analog
from gnuradio import blocks
from gnuradio.filter import firdes
from gnuradio import filter

def qpsk_awgn_generator(batch_size = 4500, EbN0s = range(-20, 10, 2)):
    data = {}
    ntaps = 45
    N_BITS = 1000000
    nvecs_per_key = batch_size
    vec_length = 128
    
    for EbN0 in EbN0s:
        data[("busy", EbN0)] = np.zeros([nvecs_per_key, 2, vec_length], dtype=np.float32)
        data[("idle", EbN0)] = np.zeros([nvecs_per_key, 2, vec_length], dtype=np.float32)

        tb = gr.top_block()
        const = digital.qpsk_constellation()
        rrc_taps = firdes.root_raised_cosine(1, 4, 1, 0.35, ntaps)
        src = blocks.vector_source_b(map(int, np.random.randint(0, const.arity(), N_BITS/const.bits_per_symbol())), False)
        
        mod = digital.chunks_to_symbols_bc((const.points()), 1)
        match_filter = filter.interp_fir_filter_ccc(4, (rrc_taps))
        amplitude = blocks.multiply_const_vcc((4, ))
        
        #noise_amplitude = 1.0 / math.sqrt(const.bits_per_symbol()* 10**(float(EbN0)/10))
        noise_amplitude = 1.0 / math.sqrt(10**(float(EbN0)/10))
        noise = analog.noise_source_c(analog.GR_GAUSSIAN, noise_amplitude, 0)
        
        add = blocks.add_vcc(1)
        sink = blocks.vector_sink_c()
        noise_sink = blocks.vector_sink_c()
        tb.connect(src, mod, match_filter, amplitude, (add, 0), sink)
        tb.connect(noise, (add, 1))
        tb.connect(noise, noise_sink)
        tb.run()

        raw_output_vector = np.array(sink.data(), dtype=np.complex64)
        raw_noise_vector = np.array(noise_sink.data(), dtype=np.complex64)

        sampler_indx = random.randint(50, 500)
        vec_indx = 0
        while sampler_indx + vec_length < len(raw_output_vector) and vec_indx < nvecs_per_key:
            sampled_vector = raw_output_vector[sampler_indx:sampler_indx+vec_length]
            data[("busy", EbN0)][vec_indx, 0,:] = np.real(sampled_vector)
            data[("busy", EbN0)][vec_indx, 1,:] = np.imag(sampled_vector)
            sampled_noise_vector = raw_noise_vector[sampler_indx:sampler_indx+vec_length]
            data[("idle", EbN0)][vec_indx, 0,:] = np.real(sampled_noise_vector)
            data[("idle", EbN0)][vec_indx, 1,:] = np.imag(sampled_noise_vector)
            sampler_indx += vec_length
            vec_indx += 1

    return data
    
if __name__ == "__main__":
    qpsk_awgn_generator()

