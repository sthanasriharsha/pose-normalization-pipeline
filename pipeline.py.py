import sys, os, glob, cv2, torch, numpy as np, json
sys.path.insert(0, '.')
from annotator.dwpose import DWposeDetector

R_SHO, L_SHO = 2, 5
NECK          = 1
BODY_KPS      = list(range(0, 18))
FACE_KPS      = list(range(24, 92))
LHAND_KPS     = list(range(92, 113))
RHAND_KPS     = list(range(113, 134))

OUT_W, OUT_H    = 512, 512
TARGET_SHO_DIST = 150
CONF            = 0.3
CROP_TOP    = 0.0   # keep from top (0% = full top)
CROP_BOTTOM = 0.75  # cut at 75% from top — removes legs, keeps thighs


def get_raw_keypoints(detector, frame):
    """
    Use DWposeDetector.pose_estimation (Wholebody) to get raw pixel coords.
    Returns kp_px (134,2) and sc (134,) exactly like DWposeDetector.__call__ does internally.
    """
    H, W = frame.shape[:2]
    with torch.no_grad():
        candidate, subset = detector.pose_estimation(frame)
        # candidate: (1, 134, 2) pixels
        # subset:    (1, 134) scores
        kp_px = candidate[0].copy()   # (134, 2) raw pixels
        sc    = subset[0].copy()      # (134,) scores
    return kp_px, sc, W, H


def normalize_frame(kp_px, sc, src_w, src_h):
    """Shoulder-based centering and scaling."""
    kp = kp_px.copy().astype(np.float64)

    r_ok = float(sc[R_SHO]) > CONF
    l_ok = float(sc[L_SHO]) > CONF

    if r_ok and l_ok:
        mid  = (kp[R_SHO] + kp[L_SHO]) / 2.0
        dist = float(np.linalg.norm(kp[R_SHO] - kp[L_SHO]))
    elif r_ok:
        mid, dist = kp[R_SHO].copy(), TARGET_SHO_DIST
    elif l_ok:
        mid, dist = kp[L_SHO].copy(), TARGET_SHO_DIST
    elif float(sc[NECK]) > CONF:
        mid, dist = kp[NECK].copy(), TARGET_SHO_DIST
    else:
        mid  = np.array([src_w / 2.0, src_h / 3.0])
        dist = TARGET_SHO_DIST

    if dist < 10:
        dist = TARGET_SHO_DIST

    scale    = TARGET_SHO_DIST / dist
    target_x = OUT_W / 2.0
    target_y = OUT_H * 0.32

    kp_norm = np.zeros_like(kp)
    for i in range(len(kp)):
        kp_norm[i][0] = (kp[i][0] - mid[0]) * scale + target_x
        kp_norm[i][1] = (kp[i][1] - mid[1]) * scale + target_y

    return kp_norm, scale


def build_pose_dict(kp_norm, sc):
    """
    Build pose dict in EXACT DWposeDetector format using normalized coords.
    Matches the expected output format:
    {
      'bodies':      {'candidate': (18,2), 'subset': (1,18), 'score': (1,18)},
      'hands':       (2, 21, 2),
      'hands_score': (2, 21),
      'faces':       (1, 68, 2),
      'faces_score': (1, 68)
    }
    All coordinates normalized to 0-1 range (x/OUT_W, y/OUT_H).
    """
    # Normalize pixel coords to 0-1
    cand = kp_norm.copy().astype(np.float64)
    cand[:, 0] /= float(OUT_W)
    cand[:, 1] /= float(OUT_H)

    # DO NOT clip — allow values > 1 for keypoints outside frame
    # (matches official DWposeDetector behavior)
    # Mark invisible keypoints as -1
    invisible = sc < CONF

    # ── Body candidate: exactly (18, 2) normalized ──
    body_cand = cand[:18].copy()
    body_cand[invisible[:18]] = -1

    # ── Body subset: (1, 18) with index or -1 ──
    body_subset = np.full((1, 18), -1.0)
    for j in range(18):
        if float(sc[j]) > CONF:
            body_subset[0][j] = float(j)

    # ── Body score: (1, 18) ──
    body_score = sc[:18].reshape(1, 18).copy()

    # ── Hands: (2, 21, 2) — [left_hand, right_hand] ──
    lhand = cand[92:113].copy()   # (21, 2)
    rhand = cand[113:134].copy()  # (21, 2)
    lhand[sc[92:113]  < CONF] = 0
    rhand[sc[113:134] < CONF] = 0
    hands = np.array([lhand, rhand])   # (2, 21, 2)

    # ── Hands score: (2, 21) ──
    hands_score = np.array([
        sc[92:113].copy(),
        sc[113:134].copy()
    ])

    # ── Faces: (1, 68, 2) ──
    face = cand[24:92].copy()   # (68, 2)
    face[sc[24:92] < CONF] = 0
    faces = face[np.newaxis, :, :]   # (1, 68, 2)

    # ── Faces score: (1, 68) ──
    faces_score = sc[24:92].reshape(1, 68).copy()

    pose_dict = {
        'bodies': {
            'candidate': body_cand,    # (18, 2)  normalized 0-1
            'subset':    body_subset,  # (1, 18)  index or -1
            'score':     body_score,   # (1, 18)  confidence
        },
        'hands':       hands,          # (2, 21, 2) normalized 0-1
        'hands_score': hands_score,    # (2, 21)
        'faces':       faces,          # (1, 68, 2) normalized 0-1
        'faces_score': faces_score,    # (1, 68)
    }

    return pose_dict


def draw_from_pose_dict(pose_dict):
    """Draw skeleton using official DWPose util functions."""
    from annotator.dwpose import util
    canvas = np.zeros((OUT_H, OUT_W, 3), dtype=np.uint8)
    canvas = util.draw_bodypose(
        canvas,
        pose_dict['bodies']['candidate'],
        pose_dict['bodies']['subset']
    )
    canvas = util.draw_handpose(canvas, pose_dict['hands'])
    canvas = util.draw_facepose(canvas, pose_dict['faces'])
    return canvas


def pose_dict_to_serializable(pose_dict, frame_idx, scale):
    """Convert numpy arrays to JSON-serializable lists."""
    def to_list(x):
        return x.tolist() if isinstance(x, np.ndarray) else x

    return {
        'frame':        frame_idx,
        'scale_factor': round(float(scale), 4),
        'bodies': {
            'candidate': to_list(pose_dict['bodies']['candidate']),
            'subset':    to_list(pose_dict['bodies']['subset']),
            'score':     to_list(pose_dict['bodies']['score']),
        },
        'hands':       to_list(pose_dict['hands']),
        'hands_score': to_list(pose_dict['hands_score']),
        'faces':       to_list(pose_dict['faces']),
        'faces_score': to_list(pose_dict['faces_score']),
    }


def process(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    print('\n' + '='*50)
    print('Processing:', name)
    print('='*50)

    detector = DWposeDetector()

    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS)
    W     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(' ', W, 'x', H, '|', fps, 'fps |', total, 'frames')

    skel_path  = os.path.join(out_dir, name + '_normalized.mp4')
    json_path  = os.path.join(out_dir, name + '_norm_kps.json')
    npy_path   = os.path.join(out_dir, name + '_norm_kps.npy')
    score_path = os.path.join(out_dir, name + '_norm_scores.npy')

    writer = cv2.VideoWriter(
        skel_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (OUT_W, OUT_H))

    all_norm, all_sc, jdata, scales = [], [], [], []
    i = 0

    # Quick format check on first frame
    first_frame_printed = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Get raw pixel keypoints via DWposeDetector
        kp_px, sc, src_w, src_h = get_raw_keypoints(detector, frame)

        # Step 2: Normalize (shoulder-based centering + scaling)
        kp_norm, scale = normalize_frame(kp_px, sc, src_w, src_h)

        # Step 3: Build pose dict in official format
        pose_dict = build_pose_dict(kp_norm, sc)

        # Step 4: Draw using official util functions
        canvas = draw_from_pose_dict(pose_dict)

        # Step 5: Crop to upper body (same crop for ALL videos)
        y1 = int(OUT_H * CROP_TOP)
        y2 = int(OUT_H * CROP_BOTTOM)
        cropped = canvas[y1:y2, :]
        final   = cv2.resize(cropped, (OUT_W, OUT_H), interpolation=cv2.INTER_LINEAR)

        writer.write(final)

        # Step 5: Store
        all_norm.append(kp_norm.copy())
        all_sc.append(sc.copy())
        scales.append(scale)
        jdata.append(pose_dict_to_serializable(pose_dict, i, scale))

        # Print format check on first frame only
        if not first_frame_printed:
            print('\n--- FORMAT CHECK (frame 0) ---')
            print('bodies.candidate shape:', pose_dict['bodies']['candidate'].shape,
                  '  dtype:', pose_dict['bodies']['candidate'].dtype)
            print('bodies.candidate[0]:', pose_dict['bodies']['candidate'][0])
            print('bodies.subset shape:', pose_dict['bodies']['subset'].shape)
            print('hands shape:', pose_dict['hands'].shape)
            print('faces shape:', pose_dict['faces'].shape)
            print('hands_score shape:', pose_dict['hands_score'].shape)
            print('faces_score shape:', pose_dict['faces_score'].shape)
            print('--- END FORMAT CHECK ---\n')
            first_frame_printed = True

        i += 1
        if i % 30 == 0 or i == total:
            print(' ['+str(i)+'/'+str(total)+'] scale='+str(round(scale, 2)))

    cap.release()
    writer.release()

    norm_arr = np.array(all_norm)   # (T, 134, 2)
    sc_arr   = np.array(all_sc)     # (T, 134)
    np.save(npy_path,   norm_arr)
    np.save(score_path, sc_arr)

    with open(json_path, 'w') as f:
        json.dump({
            'video':        name,
            'fps':          fps,
            'out_w':        OUT_W,
            'out_h':        OUT_H,
            'total_frames': i,
            'avg_scale':    round(float(np.mean(scales)), 3),
            'format':       'DWposeDetector normalized pose_dict',
            'frames':       jdata
        }, f, indent=2)

    r_sho = norm_arr[:, R_SHO, :]
    l_sho = norm_arr[:, L_SHO, :]
    sho_d = np.linalg.norm(r_sho - l_sho, axis=1)
    print('  Shoulder dist: mean='+str(round(sho_d.mean(), 1))+
          '  std='+str(round(sho_d.std(), 1))+'  target='+str(TARGET_SHO_DIST))
    print('  NPY shape:', norm_arr.shape)
    print('  DONE')


if __name__ == '__main__':

    IN  = 'input_videos'
    OUT = 'output_results'

    os.makedirs(IN, exist_ok=True)
    os.makedirs(OUT, exist_ok=True)

    videos = sorted(glob.glob(os.path.join(IN, '*.mp4')))

    if len(videos) == 0:
        print("No videos found in 'input_videos' folder.")
    else:
        print(f'Found {len(videos)} videos')

    for v in videos:
        process(v, OUT)

    print('\nALL DONE')