from liveconfig import liveclass


@liveclass
class Config:
    def __init__(self):
        self.camera_port = 1
        self.font_color = (255, 255, 255)
        self.font_scale = 1.0
        self.font_thickness = 2
        # Order: red, white, yellow, green, blue, brown, pink, black
        self.bbox_colors = [
            (0, 0, 255),      # red
            (255, 255, 255),  # white
            (0, 255, 255),    # yellow
            (0, 255, 0),      # green
            (255, 0, 0),      # blue (BGR)
            (42, 42, 165),    # brown
            (203, 192, 255),  # pink
            (0, 0, 0)         # black
        ]
        self.use_obstruction_detection = False  # Disabled - no autoencoder model yet
        self.autoencoder_model_path = "./tests/src/data/model/autoencoder_model.keras"
        self.collect_ae_data = False
        self.ae_data_path = "./tests/src/data/model/ae_data/"
        self.obstruction_threshold = 0.013
        self.obstruction_warn_if_within = 0.05
        self.obstruction_buffer_size = 4
        self.use_networking = False  # Disabled - no server to connect to
        self.network_update_interval = 0.1
        self.poolpal_url = "http://poolpal.joshn.uk"
        self.output_dimensions = (1200, 600)
        self.ball_area_range = (1500, 4500)
        self.arm_area_range = (10000, 20000)
        self.gantry_effective_range_x_px = (100, 1100)
        self.gantry_effective_range_y_px = (84, 516)
        self.use_hidden_balls = False
        self.use_model = True
        self.process_every_n_frames = 2
        self.detection_model_path = "final/src/snookervision/data/model/best_color.pt"  
        self.position_threshold = 6
        self.hole_threshold = 30
        self.conf_threshold = 0.5
        self.draw_results = True
        self.hide_windows = False
        self.use_calibration = False  # Disabled - causes zoom issues
        self.use_table_pts = False  # Disabled - select manually on first run
        self.model_image_path = "./tests/src/data/model/training_images/"
        self.collect_model_images = False
        self.calibration_params_path = "./tests/src/data/calibration_params.json"
        self.calibration_images_path = "./tests/src/data/calibration_images/"
        self.table_pts_path = "./tests/src/data/table_pts.json"
