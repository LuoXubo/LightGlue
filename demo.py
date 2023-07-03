from lightglue import LightGlue, SuperPoint
from lightglue.utils import load_image, match_pair
from lightglue import viz2d
from pathlib import Path
import torch
images = Path('assets')

extractor = SuperPoint(max_num_keypoints=2048, nms_radius=3).eval().cuda()  # load the extractor
match_conf = {
    'width_confidence': 0.99,  # for point pruning
    'depth_confidence': 0.95,  # for early stopping,
}
matcher = LightGlue(pretrained='superpoint', **match_conf).eval().cuda()

# image0, scales0 = load_image(images / 'DSC_0411.JPG', resize=1024, grayscale=False)
# image1, scales1 = load_image(images / 'DSC_0410.JPG', resize=1024, grayscale=False)

image0, sc0 = load_image(images / 'sacre_coeur1.jpg', resize=1024, grayscale=False)
image1, sc1 = load_image(images / 'sacre_coeur2.jpg', resize=1024, grayscale=False)

pred = match_pair(extractor, matcher, image0, image1)

kpts0, kpts1, matches = pred['keypoints0'], pred['keypoints1'], pred['matches']
m_kpts0, m_kpts1 = kpts0[matches[..., 0]], kpts1[matches[..., 1]]

axes = viz2d.plot_images([image0.permute(1, 2, 0), image1.permute(1, 2, 0)])
viz2d.plot_matches(m_kpts0, m_kpts1, color='lime', lw=0.2)
viz2d.add_text(0, f'Stop after {pred["stop"]} layers', fs=20)
viz2d.save_plot('matches.png')

kpc0, kpc1 = viz2d.cm_prune(pred['prune0']), viz2d.cm_prune(pred['prune1'])
viz2d.plot_images([image0.permute(1, 2, 0), image1.permute(1, 2, 0)])
viz2d.plot_keypoints([kpts0, kpts1], colors=[kpc0, kpc1], ps=10)
viz2d.save_plot('kpts.png')
print(len(kpts0))
