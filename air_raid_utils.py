import numpy as np
from collections import deque

# This is specific to the Airraid environment
def img_crop(img):
    return img[20:230,:,:]

# GENERAL Atari preprocessing steps
def downsample(img):
    # We will take only half of the image resolution
    return img[::2, ::2]

def transform_reward(reward):
    return np.sign(reward)

def to_grayscale(img):
    return np.mean(img, axis=2).astype(np.uint8)

# Normalize grayscale image from -1 to 1.
def normalize_grayscale(img):
    return (img - 128) / 128 - 1  

def process_frame(img):
    #print("Inside AirRaid process frame")
    img = img_crop(img)
    img = downsample(img)    # Crop and downsize (by 2)
    img = to_grayscale(img)       # Convert to greyscale by averaging the RGB values
    img = normalize_grayscale(img)  # Normalize from -1 to 1.
    return np.expand_dims(img, axis=-1)

def get_stacked_frames(frames, new_frame, is_new_episode, stack_size=4):
    new_frame = process_frame(new_frame)
    
    if is_new_episode:
        frames = deque([new_frame] * stack_size, maxlen=stack_size)
    else:
        frames.append(new_frame)
        
    # Stack frames along the last dimension
    stacked_frames = np.concatenate(list(frames), axis=-1)
    # Add batch dimension
    stacked_frames = np.expand_dims(stacked_frames, axis=0)
    return stacked_frames, frames

# OTHER
class RingBuf:
    def __init__(self, size):
        # Pro-tip: when implementing a ring buffer, always allocate one extra element,
        # this way, self.start == self.end always means the buffer is EMPTY, whereas
        # if you allocate exactly the right number of elements, it could also mean
        # the buffer is full. This greatly simplifies the rest of the code.
        self.data = [None] * (size + 1)
        self.start = 0
        self.end = 0
        
    def append(self, element):
        self.data[self.end] = element
        self.end = (self.end + 1) % len(self.data)
        # end == start and yet we just added one element. This means the buffer has one
        # too many element. Remove the first element by incrementing start.
        if self.end == self.start:
            self.start = (self.start + 1) % len(self.data)
        
    def __getitem__(self, idx):
        return self.data[(self.start + idx) % len(self.data)]
    
    def __len__(self):
        if self.end < self.start:
            return self.end + len(self.data) - self.start
        else:
            return self.end - self.start
        
    def __iter__(self):
        for i in range(len(self)):
            yield self[i]