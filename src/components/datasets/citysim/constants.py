_DISPARITY_LIMIT_= 24
_BASELINE_ = 0.1
_LOCATIONS_ = {'left'}
FRAMES_FILTER = {
    'town1':(50, 1500),
    'town2_night':(50, 1500),
    'town3':(50, 1500),
    'town4':(25, 1500),
    'town5_night':(25, 1500),

}

# Training_set = ['town1']
Training_set = ['town1', 'town2_night', 'town3']

# Testing_set = ['town4', 'town5_night']
Testing_set = [ 'town5_night']
# Testing_set = ['town4']
# Testing_set = ['town1']