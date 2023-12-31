from configs import transforms_config
from configs.paths_config import dataset_paths


DATASETS = {
	'ffhq_encode': {
		'transforms': transforms_config.EncodeTransforms,
		'train_source_root': dataset_paths['ffhq_source'],
		'train_target_root': dataset_paths['ffhq_target'],
		'test_source_root': dataset_paths['ffhq_test_source'],
		'test_target_root': dataset_paths['ffhq_test_target'],
	},
	'render_encode': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['render_train_source'],
		'train_target_root': dataset_paths['render_train_target'],
		'test_source_root': dataset_paths['render_test_source'],
		'test_target_root': dataset_paths['render_test_target'],
	},
	'render_encode_5k': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['render_train_source_5k'],
		'train_target_root': dataset_paths['render_train_target_5k'],
		'test_source_root': dataset_paths['render_test_source_5k'],
		'test_target_root': dataset_paths['render_test_target_5k'],
	},
	'render_encode_5k_8d': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['render_train_source_5k_8d'],
		'train_target_root': dataset_paths['render_train_target_5k_8d'],
		'test_source_root': dataset_paths['render_test_source_5k_8d'],
		'test_target_root': dataset_paths['render_test_target_5k_8d'],
	},
	'wild_1650': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['wild_1650_train_source'],
		'train_target_root': dataset_paths['wild_1650_train_target'],
		'test_source_root': dataset_paths['wild_1650_test_source'],
		'test_target_root': dataset_paths['wild_1650_test_target'],
	},
	'eye_1386': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['eye_1386_train_source'],
		'train_target_root': dataset_paths['eye_1386_train_target'],
		'test_source_root': dataset_paths['eye_1386_test_source'],
		'test_target_root': dataset_paths['eye_1386_test_target'],
	},
	'back_1386': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['back_1386_train_source'],
		'train_target_root': dataset_paths['back_1386_train_target'],
		'test_source_root': dataset_paths['back_1386_test_source'],
		'test_target_root': dataset_paths['back_1386_test_target'],
	},
	'back_4592': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['back_4592_train_source'],
		'train_target_root': dataset_paths['back_4592_train_target'],
		'test_source_root': dataset_paths['back_4592_test_source'],
		'test_target_root': dataset_paths['back_4592_test_target'],
	},
	'head_global_4592': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['back_4592_train_source'],
		'train_target_root': dataset_paths['head_4592_train_target'],
		'test_source_root': dataset_paths['back_4592_test_source'],
		'test_target_root': dataset_paths['head_4592_test_target'],
	},
	'head_single_4592': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['head_4592_train_source'],
		'train_target_root': dataset_paths['head_4592_train_target'],
		'test_source_root': dataset_paths['head_4592_test_source'],
		'test_target_root': dataset_paths['head_4592_test_target'],
	},
	'nose_single_4592': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['nose_4592_train_source'],
		'train_target_root': dataset_paths['nose_4592_train_target'],
		'test_source_root': dataset_paths['nose_4592_test_source'],
		'test_target_root': dataset_paths['nose_4592_test_target'],
	},
	'mouth_single_4592': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['mouth_4592_train_source'],
		'train_target_root': dataset_paths['mouth_4592_train_target'],
		'test_source_root': dataset_paths['mouth_4592_test_source'],
		'test_target_root': dataset_paths['mouth_4592_test_target'],
	},
	'body_single_4592': {
		'transforms': transforms_config.RenderTransforms,
		'train_source_root': dataset_paths['body_4592_train_source'],
		'train_target_root': dataset_paths['body_4592_train_target'],
		'test_source_root': dataset_paths['body_4592_test_source'],
		'test_target_root': dataset_paths['body_4592_test_target'],
	},
	'ffhq_frontalize': {
		'transforms': transforms_config.FrontalizationTransforms,
		'train_source_root': dataset_paths['ffhq'],
		'train_target_root': dataset_paths['ffhq'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celebs_sketch_to_face': {
		'transforms': transforms_config.SketchToImageTransforms,
		'train_source_root': dataset_paths['celeba_train_sketch'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test_sketch'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celebs_seg_to_face': {
		'transforms': transforms_config.SegToImageTransforms,
		'train_source_root': dataset_paths['ffhq_source'],
		'train_target_root': dataset_paths['ffhq_target'],
		'test_source_root': dataset_paths['ffhq_test_source'],
		'test_target_root': dataset_paths['ffhq_test_target'],
	},
	'celebs_super_resolution': {
		'transforms': transforms_config.SuperResTransforms,
		'train_source_root': dataset_paths['celeba_train'],
		'train_target_root': dataset_paths['celeba_train'],
		'test_source_root': dataset_paths['celeba_test'],
		'test_target_root': dataset_paths['celeba_test'],
	},
	'celebs_sketch_to_texture': {
		'transforms': transforms_config.SketchToImageTransforms,
		'train_source_root': dataset_paths['ffhq_source'],
		'train_target_root': dataset_paths['ffhq_target'],
		'test_source_root': dataset_paths['ffhq_test_source'],
		'test_target_root': dataset_paths['ffhq_test_target'],
	},
}
