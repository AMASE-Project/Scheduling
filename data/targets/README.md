**FITS files with the following target information:**
  * Project ID: the ID in each science project; long integer.
  * Target name: the unique name in the sample file; long integer.
  * Pointing RA: Right Ascension of pointing; double precision; in unit of degree (J2000)
  * Pointing DEC: Declination of pointing; double precision; in unit of degree (J2000)
  * Exptime: Time required per exposure; double precision; in units of seconds. This value can be estimated from the Exposure Time Calculator(ETC).
  * NumExp: Number of exposures required per dither position; integer
  * Dither: Number of dithering required; integer 1, 3, 9, or 27
  * Telescope: Which telescope to use: “0.8m” or “0.14m”.
  * PointingID: For targets that require multiple tiles, the ID of the pointing that belongs to the same target. long integer
  * Priority: Priority for the pointing: From 1 to 3, with 1 being the highest and 3 being the lowest.
  * Source: from which science program.
