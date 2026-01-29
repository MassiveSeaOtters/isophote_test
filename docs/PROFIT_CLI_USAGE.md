
./profit-cli: utility program to generate an image out of a model and a set of profiles

This program is licensed under the GPLv3 license.

Usage: ./profit-cli [options] -p <spec> [-p <spec> ...]

Options:
  -t        Output image as text values on stdout
  -f <file> Output image as fits file
  -i <n>    Output performance information after evaluating the model n times
  -s        Show runtime stats
  -T <conv> Use this type of convolver (see below)
  -u        Return an un-cropped image from the convolver
  -C <p,d>  Use OpenCL with platform p, device d, and double support (0|1)
  -c        Display OpenCL information about devices and platforms
  -n <n>    Use n OpenMP threads to calculate profiles
  -e <n>    FFTW plans created with n effort (more takes longer)
  -I <n>    SIMD Instruction set to use with brute-force convolver.
            0=auto (default), 1=none, 2=sse2, 3=avx.
  -r        Reuse FFT-transformed PSF across evaluations (if -T fft)
  -x        Image width. Defaults to 100
  -y        Image height. Defaults to 100
  -S <n>    Finesampling factor. Defaults to 1
  -F        Do *not* return finesampled image (if -S <n>)
  -w        Width in pixels. Defaults to 100
  -H        Height in pixels. Defaults to 100
  -m        Zero magnitude. Defaults to 0
  -P        PSF function (specified as w:h:val1,val2..., or as a FITS filename)
  -R        Clear libprofit's cache and exit
  -h,-?     Show this help and exit
  -V        Show the program version and exit

The following convolver types are supported:

 * brute: A brute-force convolver
 * brute-old: An older, slower brute-force convolver (used only for comparisons)
 * opencl: An OpenCL-based brute-force convolver
 * fft: An FFT-based convolver

Profiles should be specified as follows (parts between [] are optional):

-p [count,]name[:param1=val1:param2=val2:...]

Here "count" specifies the number of times to repeat a given profile (useful for
scalability checks), "name" is the name of the profile (see below), and the rest
are the list of parameter and values for that profile.

The following profiles and parameters are currently supported:

 * null
 * psf: xcen, ycen, mag
 * sky: bg
 * sersic: re, nser, rescale_flux
 * moffat: fwhm, con
 * ferrer: a, b, rout
 * coresersic: re, nser, rb, a, b
 * brokenexp: h1, h2, rb, a
 * king: rc, rt, a
 * sersic, moffat, ferrer, coresersic, king: xcen, ycen, mag, box, ang, axrat,
                           rough, rscale_switch, max_recursions,
                           resolution, acc, rscale_max, adjust

For more information visit https://libprofit.readthedocs.io.

