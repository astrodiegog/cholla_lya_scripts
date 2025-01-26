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


However, the problem is that in the subroutine ``Grid3D::Compute_Flux_Power_Spectrum_Skewer``, the flux deviation from the mean is calculated as

```cpp
delta_F[los_id] = skewers_transmitted_flux[skewer_id * n_los + los_id] / Analysis.Flux_mean_HI;
```

The array is filled up in the subroutine ``Grid3D::Compute_Transmitted_Flux_Skewer``, as the following 

```cpp
skewers_transmitted_flux_HI[skewer_id * n_los_total + los_id]   = exp(-full_optical_depth_HI[los_id + n_ghost]);
```

So from my interpretation, this reads as

$$
    \delta_F (u) = \frac{F(u)}{\bar{F}}
$$

which doesn't sound right. So far in my python scripts, I have been doing the same thing... something something, if it aint broke... But I just noticed how this may be effecting the transmitted flux power spectrum. 

The goal of this study is to document the difference that results from this change from the input into the FFT. The output will be three plots:

1. Dimensionles transmitted flux power spectra $\Delta_F^2 (k)$
2. Relative error of transmitted flux power spectra with respect to analysis output $\Delta_F^2 (k)$
3. Absolute relative error of transmitted flux power spectra with respect to analysis output $\Delta_F^2 (k)$ in logspace





