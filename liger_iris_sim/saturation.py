def compute_saturation(image, wave=None, gain, out_savesim = out_savesim, n_pixels = n_pixels, out_image = out_image, out_DNimg = out_DNimg, out_badp = out_badp
  if n_elements(image) ne 0 then begin
    if n_elements(sat_phot) eq 0 then sat_phot = 45000.
    if n_elements(phot_trunc) eq 0 then phot_trunc = 100000.
    if n_elements(phot_e_gain) eq 0 then phot_e_gain =1.
    if n_elements(e_adu_gain) eq 0 then e_adu_gain = 3.04
    s=size(image)
    if s[0] eq 3 then begin
      
      bad_p=make_array(s[1],s[2],s[3])
      DNimg=make_array(s[1],s[2],s[3])
      totalObservedCube = image*e_adu_gain
      copyImg=totalObservedCube
      DNimg=totalObservedCube/e_adu_gain
      if max(totalObservedCube) gt sat_phot then begin
        sarr1 = totalObservedCube gt phot_trunc
        ind1 = where(sarr1,/null)
        totalObservedCube[ind1]= phot_trunc
        sarr2=copyImg gt sat_phot
        ind2=where(sarr2,/null)
        bad_p[ind2]=9
        copyImg[ind2]= sat_phot
        DNimg=totalObservedCube/e_adu_gain
        n_pixels=float(n_elements(ind2))
      endif else print, 'No saturation in cube detected.'
      if n_elements(out_savesim) ne 0 then begin
        fitsParams = ['Fits Extension', 'Non-linearity Point', 'Saturation Point',$
          'Photon-Electron Gain', 'Electron-ADU Gain']
        fitsValues = ['Saturated Image',scomp([sat_phot, phot_trunc, phot_e_gain, e_adu_gain])]
        mkosiriscube, wave, transpose(totalObservedCube, [2, 1, 0]), out_savesim,$
           /micron, units = 'phot', params = fitsParams, values = $
           fitsValues, /append
        fitsParams = ['Fits Extension','Units', 'Non-linearity Point', 'Saturation Point',$
           'Photon-Electron Gain', 'Electron-ADU Gain']
        fitsValues = ['Saturated Image','ADU',scomp([sat_phot, phot_trunc, phot_e_gain, e_adu_gain])]
        mkosiriscube, wave, transpose(DNimg, [2, 1, 0]), out_savesim, /micron,$
           units = 'phot', params = fitsParams, values = $
           fitsValues, /append
        fitsParams = ['Fits Extension', 'Non-linearity Point', 'Saturation Point',$
           'Photon-Electron Gain', 'Electron-Adu Gain']
        fitsValues = ['Bad Pixel Map',scomp([sat_phot, phot_trunc, phot_e_gain, e_adu_gain])]
        mkosiriscube, wave, transpose(bad_p, [2, 1, 0]), out_savesim, /micron,$
           units = 'phot', params = fitsParams, values = $
           fitsValues, /append
      endif
    endif else if s[0] eq 2 then begin
      bad_p=make_array(s[1],s[2])
      DNimg=make_array(s[1],s[2])
      copyImg=image
      DNimg=image/e_adu_gain
      if max(image) gt sat_phot then begin
        print, 'Saturation Detected!'
        sarr1 = image gt phot_trunc
        ind1 = where(sarr1,/null)
        image[ind1]=phot_trunc
        sarr2=copyImg gt sat_phot
        ind2=where(sarr2,/null)
        bad_p[ind2]=2
        print, 'Total Saturated Ratio: ', total(image,/double)/total(copyImg,/double)
        copyImg[ind2]=sat_phot
        DNimg=image/e_adu_gain
        n_pixels=float(n_elements(ind2))
      endif else print, 'No Saturation Detected.'
      if n_elements(out_savesim) ne 0 then begin
        sxaddpar, header1, 'XTENSION', 'Saturated Image' 
        sxaddpar, header1, 'NAXIS', 2
        sxaddpar, header1, 'NAXIS1', s[1]
        sxaddpar, header1, 'NAXIS2', s[2]
        sxaddpar, header1, 'UNITS', 'electrons' 
        sxaddpar, header1, 'NON-LIN', sat_phot
        sxaddpar, header1, 'SAT-VAL', phot_trunc
        sxaddpar, header1, 'PHO-EL-G', phot_e_gain
        sxaddpar, header1, 'EL-ADU-G', e_adu_gain
        sxaddpar, header1, 'CUNIT1', 'deg'
        sxaddpar, header1, 'CUNIT2', 'deg'
        sxaddpar, header1, 'CRPIX1', s[1]/2
        sxaddpar, header1, 'CRPIX2', s[2]/2
        header2 = header1
        sxaddpar, header2, 'UNITS', 'ADU' 
        sxaddpar, header3, 'XTENSION', 'DQ'
        sxaddpar, header3, 'NAXIS', 2
        sxaddpar, header3, 'NAXIS1', s[1]
        sxaddpar, header3, 'NAXIS2', s[2]
        print, 'saving', out_savesim
        print, 'appending data'
        writefits, out_savesim, image, header1, /append
        writefits, out_savesim, DNimg, header2, /append
        writefits, out_savesim, bad_p, header3, /append
        ;writefits, '/data/group/data/iris/dev/bad_pixel_map.fits', bad_p, header3, /append
      endif
      out_image = image
      out_DNimg = DNimg
      out_badp = bad_p
    endif else print, 'Image is not a 2D image or data cube.'
  endif else print, 'No image detected.'
  
end                     