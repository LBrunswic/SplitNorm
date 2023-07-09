HPset = [

    # {
    #     'EPOCHS':100,
    #     'infra_command' : 16,
    #     'batch_size' : 32,
    #     'delta' : 5*1e-1,
    # },
    # {
    #     'EPOCHS':200,
    #     'lvl1_command' : 13,
    #     'channel1_dim' : 7,
    #     'batch_size' :  128,
    #     'delta' : 5*1e-1,
    # },
    {
        'EPOCHS' : 200,
        'STEP_PER_EPOCH' : 30,
        'flow_depth' : 4,
        'flow_width' : 64,


        'channel1_width' : 32,
        'channel1_depth' : 2,
        'channel1_dim' : 2,

        'channel2_width' : 64,
        'channel2_depth' : 4,
        'channel2_sample' : 1,
        'channel2_dim' : 32,


        'batch_size' : 32,
    }
]*10
