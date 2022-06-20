class EyeSample:
    def __init__(self, orig_image, image, is_left, transform_inv, estimated_radius):
        self._orig_image = orig_image.copy()
        self._image = image.copy()
        self._is_left = is_left
        self._transform_inv = transform_inv
        self._estimated_radius = estimated_radius

    @property
    def orig_image(self):
        return self._orig_image

    @property
    def image(self):
        return _self.image

    @property
    def is_left(self):
        return self._is_left

    @property
    def transform_inv(self):
        return self._transform_inv

    @property
    def estimated_radius(self):
        return self._estimated_radius