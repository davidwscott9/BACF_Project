def resp_newton(response, responsef, iterations, ky, kx, use_sz):

    import numpy as np
    import math as m
    max_resp_row = np.amax(response, axis=0)
    max_row = np.argmax(response, axis=0)
    init_max_response = np.amax(max_resp_row, axis=0)
    max_col = np.argmax(max_resp_row, axis=0)
    col = max_col
    col = col + 1  # match MATLAB indexing for now

    row = np.zeros(len(col))
    for i in range(0, len(col)):
        row[i] = max_row[col[i]-1, i]
    row = row + 1  # match MATLAB indexing for now

    trans_row = (row - 1 + np.floor((use_sz[0] - 1) / 2)) % use_sz[0] - np.floor((use_sz[0] - 1) / 2)
    trans_col = (col - 1 + np.floor((use_sz[1] - 1) / 2)) % use_sz[1] - np.floor((use_sz[1] - 1) / 2)

    init_pos_y = 2 * m.pi * trans_row / use_sz[0]
    init_pos_x = 2 * m.pi * trans_col / use_sz[1]
    max_pos_y = init_pos_y
    max_pos_x = init_pos_x

    # pre-compute complex exponential
    exp_iky = np.exp(np.multiply(1j * ky.reshape(1,-1)[:,:,None], max_pos_y.reshape(1,1,-1)))
    exp_ikx = np.exp(np.multiply(1j * kx.reshape(-1,1)[:,:,None], max_pos_x.reshape(1,1,-1)))

    ky2 = np.multiply(ky, ky)
    kx2 = np.multiply(kx, kx)  # --> SAME AS MATLAB


    counter = 1
    while counter <= iterations:
        ky_exp_ky = np.multiply(ky.reshape(1,-1)[:,:,None], exp_iky)
        kx_exp_kx = np.multiply(kx.reshape(-1,1)[:,:,None], exp_ikx)
        y_resp = np.einsum('mnr,ndr->mdr', exp_iky, responsef)  # use this instead of mtimesx
        resp_x = np.einsum('mnr,ndr->mdr', responsef, exp_ikx)
        grad_y = -np.imag(np.einsum('mnr,ndr->mdr', ky_exp_ky, resp_x))
        grad_x = -np.imag(np.einsum('mnr,ndr->mdr', y_resp, kx_exp_kx))
        ival = 1j * np.einsum('mnr,ndr->mdr', exp_iky, resp_x)
        H_yy = np.real(-1*np.einsum('mnr,ndr->mdr', np.multiply(ky2.reshape(1,-1)[:,:,None], exp_iky), resp_x) + ival)
        H_xx = np.real(-1*np.einsum('mnr,ndr->mdr', y_resp, np.multiply(kx2.reshape(-1,1)[:,:,None], exp_ikx)) + ival)
        H_xy = np.real(-1*np.einsum('mnr,ndr->mdr', ky_exp_ky, np.einsum('mnr,ndr->mdr', responsef, kx_exp_kx)))
        det_H = np.multiply(H_yy, H_xx) - np.multiply(H_xy, H_xy)

        # compute new position using newtons method
        max_pos_y = max_pos_y - np.divide((np.multiply(H_xx, grad_y) - np.multiply(H_xy, grad_x)), det_H)
        max_pos_x = max_pos_x - np.divide((np.multiply(H_yy, grad_x) - np.multiply(H_xy, grad_y)), det_H)

        # Evaluate maximum
        exp_iky = np.exp(np.multiply(1j * ky.reshape(1,-1)[:,:,None], max_pos_y))
        exp_ikx = np.exp(np.multiply(1j * kx.reshape(-1,1)[:,:,None], max_pos_x))

        counter += 1

    max_response = 1 / np.prod(use_sz) * np.real(np.einsum('mnr,ndr->mdr',
                                                           np.einsum('mnr,ndr->mdr', exp_iky, responsef), exp_ikx))

    # check for scales that have not increased in score
    ind = max_response[0,0,:] < init_max_response
    max_response[0,0,ind] = init_max_response[ind]
    max_pos_y[0,0,ind] = init_pos_y[ind]
    max_pos_x[0,0,ind] = init_pos_x[ind]

    sind = np.argmax(max_response)
    disp_row = ((max_pos_y[0, 0, sind] + m.pi) % (2 * m.pi) - m.pi) / (2 * m.pi) * use_sz[0]
    disp_col = ((max_pos_x[0, 0, sind] + m.pi) % (2 * m.pi) - m.pi) / (2 * m.pi) * use_sz[1]
    #  --> ALL MATCHES MATLAB NOW :)

    return disp_row, disp_col, sind
