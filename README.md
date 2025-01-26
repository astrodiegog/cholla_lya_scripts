# cholla_lya_scripts

Python scripts to study the Lyman-alpha Forest in cosmological Cholla simulations

I have recently seen that in Cholla, the transmitted flux power spectrum is calculated in a bit of an odd way.

To calculate the transmitted flux power spectrum, we first calculate the flux deviations from the mean

$$
    \delta_F (u) = \frac{F(u) - \bar{F}}{\bar{F}}
$$

where $u$ is the velocity.

We then take the Fourier Transform as 

$$
    \tilde{\delta}_F(k) = \frac{1}{u_{\textrm{max}}} \int_0^{u_{\textrm{max}}} e^{-iku} \delta_F (u) \textrm{d}u
$$

where $u_{\textrm{max}}$ is the maximum peculiar velocity across the entire box. In this case, it corresponds to the Hubble flow across the entire box length.

We then take the power and average to find

$$r
    P(k) = u_{\textrm{max}} \left\langle \left| \tilde{\delta}_F(k) \right|^2 \right\rangle
$$

In literature, the commonly accepted description is the dimensionless transmitted flux power spectrum.

$$
    \Delta_F^2 (k) = \frac{1}{\pi} k P(k)
$$

