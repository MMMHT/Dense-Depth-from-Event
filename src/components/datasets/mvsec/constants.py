_LOCATIONS_ = {'left'}


_DISPARITY_LIMIT_ = {'indoor_flying1': 32,
                     'indoor_flying2': 32,
                     'indoor_flying3': 32,
                     'outdoor_day1': 12,
                     'outdoor_day2': 12,
                     'outdoor_night1': 12,
                     'outdoor_night2': 12,
                     'outdoor_night3': 12,
                     }

Training_set = {'outdoor_day2'}
Testing_set = {'outdoor_day1'}

FRAMES_FILTER = {
    'indoor_flying': {
    'train':{
        '1': (80, 1260),
        '2': (160, 1580),
        '3': (125, 1815),
        '4': (190, 290)
        },
    'test':{
           '1': (140, 1201),
           '2': (120, 1421),
           '3': (73, 1616),
           '4': (190, 290)
       },
    'val':{
            '1':(300, 600),
           '2': (120, 1421),
           '3': (73, 1616),
           '4': (190, 290)
       },
    },
'outdoor_day': {
    'train':{
        '1': (500, 5135),
        '2': (500, 12000),
        },
    'test':{
           '1': (25, 5135),
           '2': (25, 12190),
       },
    'val':{
           '1':(1000, 1500),
           '2':(3000, 3500),
       },
    },
'outdoor_night': {
    'train':{
        '1': (50, 5133),
        '2': (50, 5497),
        '3': (50, 5429),
        },
    'test':{
           '1': (25, 5133),
           '2': (25, 5497),
           '3': (25, 5429),
       },
    'val':{
            '1':(100, 400),
            '2': (100, 400),
            '3': (100, 400),
       },
    }
}

