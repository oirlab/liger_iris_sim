def generate_filename(
        obsid : str,
        detector : str, obstype : str, level : int | str = 0,
        exp : int | str = '0001', subarray : int | str | None = None,
    ):
    if isinstance(exp, int):
        exp = str(exp).zfill(4)
    if type(subarray) in (int, str):
        subarray = '-' + str(subarray).zfill(2)
    else:
        subarray = '-00'
    return f"{obsid}_IRIS_{detector.upper()}_{obstype}_LVL{int(level)}_{exp}{subarray}.fits"