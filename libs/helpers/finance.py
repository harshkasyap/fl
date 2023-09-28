def bin_hours_per_week(hpw):
    """
    Bin the hours per week variable into 0-30, 31-40, 41-50, 50+ bins.
    """
    if hpw <= 30:
        return 0
    elif hpw <= 40:
        return 1
    elif hpw <= 50:
        return 2
    return 3


def bin_edu_level(edu_level):
    """
    Bin the hours per week variable into 0-30, 31-40, 41-50, 50+ bins.
    """
    if edu_level < 2:
        return 0
    elif edu_level < 4:
        return 1
    else:
        return 2

def bin_age_level(edu_level):
    """
    Bin the hours per week variable into 0-30, 31-40, 41-50, 50+ bins.
    """
    if edu_level < 5:
        return 0
    elif edu_level < 10:
        return 1
    elif edu_level < 14:
        return 2
    else:
        return 3
    
def bin_marital_status_level(edu_level):
    return edu_level

def bin_NATIVITY_level(edu_level):
    return edu_level

def test_SEX_enum(edu_level):
    return edu_level

def test_RACIP_enum(edu_level):
    if edu_level < 2:
        return 1
    else:
        return 2
