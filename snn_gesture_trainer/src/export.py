def export_spike_data(spike_data, out_file):
    """
    Converts spike encoded data to FENRIR readable binary data.

    Args:
        spike_data: spike encoded data of shape [tsteps, 1, x, y]
        out_file: path to output file
    """
    exp_data    = spike_data[:, 0, :, :]
    exp_data    = exp_data.view(exp_data.shape[0], -1)
    tsteps      = exp_data.shape[0]
    events      = []
    tstep_event_idx = []

    for t in range(0, tsteps):
        t_data = exp_data[t, :]
        non_zero_indices = (t_data != 0).nonzero(as_tuple=True)[0]

        for idx in non_zero_indices.tolist():
            events.append(idx)

        events.append(0b1000000000000)

        tstep_event_idx.append(len(events))

    binary_events = [format(idx, '010b') for idx in events]

    with open(out_file, 'w', encoding='utf-8') as f:
        for b in binary_events:
            if not b == '1000000000000':
                f.write("000" + b + '\n')
            else:
                f.write(b + '\n')

def export_weights(fc, bits_per_addr, input_size, out_file) -> None:
    """
    Exports the quantized weights to a file.

    Args:
        fc: fc layer e.g. mymodel.fc1
        bits_per_addr: number of bits per synapse address
        input_size: input layer size e.g. 32x32=1024
        out_file: path to output file
    """

    scaled_weights = fc.quant_weight()/fc.quant_weight().scale
    scaled_weights = scaled_weights.cpu().detach().numpy()

    qsyn_bin = []
    for in_nrn in range(0, input_size):
        syn_weights = scaled_weights[:, in_nrn]
        for syn in syn_weights:
            qsyn = int(round(syn))
            qsyn = max(-8, min(7, qsyn))
            qsyn_bin.append(format(qsyn & 0b1111, '04b'))

    str_var = ""
    lines = []
    for syn in qsyn_bin:
        str_var += syn
        if len(str_var) == bits_per_addr:
            lines.append(str_var)
            str_var = ""

    with open(out_file, "w") as f:
        for line in lines:
            f.write(line + "\n")

def get_threshold(fc, lif) -> float:
    """
    Returns the scaled LIF threshold.

    Args:
        lif: lif layer e.g. mymodel.lif1
    """    
    scaled_thr = lif.threshold/fc.quant_weight().scale
    scaled_thr = scaled_thr.cpu().detach().numpy()
    return scaled_thr