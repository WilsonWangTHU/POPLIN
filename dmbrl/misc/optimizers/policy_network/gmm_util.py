# -----------------------------------------------------------------------------
#   @brief:
# -----------------------------------------------------------------------------

import numpy as np


def get_conditional_gaussian(mean, cov, observation_size):
    """ @brief: see the function with the same name in mbbl

        y = f_c + f_d.dot(x)
        cov(y) = pi_cov
    """

    condition_size = observation_size
    pi_x = np.linalg.solve(cov[:condition_size, :condition_size],
                           cov[:condition_size, condition_size:]).T
    pi_c = mean[condition_size:] - pi_x.dot(mean[:condition_size])
    pi_cov = cov[condition_size:, condition_size:] - \
        pi_x.dot(cov[:condition_size, :condition_size]).dot(pi_x.T)
    pi_cov = 0.5 * (pi_cov + pi_cov.T)

    # return {'pol_k': pi_c, 'pol_K': pi_x, 'pol_S': pi_cov}
    return {'f_c': pi_c, 'f_d': pi_x, 'cov': pi_cov}


def get_gmm_posterior(gmm, gmm_weights, data):
    """ @brief: see the function with the same name in mbbl
    """

    # posterior mean of gmm (C --> num_cluster, N --> num_data)
    response = gmm.predict_proba(np.reshape(data, [1, -1]))  # (N, C)
    # (C, 1)
    avg_response = np.reshape(np.mean(np.array(response), axis=0), [-1, 1])
    pos_mean = np.mean(avg_response * gmm_weights['mean'], axis=0)  # (Vec)

    # posterior cov = (sum_i) res_i * (cov_i + \mu_i(\mu_i - \mu)^T)
    diff_mu = gmm_weights['mean'] - np.expand_dims(pos_mean, axis=0)  # (C, Vec)
    mui_mui_muT = np.expand_dims(gmm_weights['mean'], axis=1) * \
        np.expand_dims(diff_mu, axis=2)  # (C, Vec, Vec), the outer product
    response_expand = np.expand_dims(avg_response, axis=2)
    pos_cov = np.sum((gmm_weights['cov'] + mui_mui_muT) *
                     response_expand, axis=0)

    return pos_mean, pos_cov
