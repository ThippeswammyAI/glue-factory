Phase 1: START WITH A BASIC DETECTOR
═════════════════════════════════════

BEFORE Homographic Adaptation:
┌──────────────────────────────┐
│ Use EXISTING detector:       │
│ - SIFT (hand-crafted)        │
│ - ORB (hand-crafted)         │
│ - FAST (hand-crafted)        │
│ - MagicPoint (pretrained)    │
│                              │
│ These detectors find initial │
│ keypoints (even if not great)│
│ Maybe 50-100 keypoints       │
└──────────────────────────────┘
           ↓

Phase 2: HOMOGRAPHIC ADAPTATION LOOP
═════════════════════════════════════

For each training iteration:

1️⃣  Take image I
    ↓
2️⃣  Apply random homography H → get I'
    ↓
3️⃣  Use BASIC DETECTOR on BOTH images:
    ├─ Detect in I → keypoints_I
    └─ Detect in I' → keypoints_I'
    ↓
4️⃣  Match keypoints via homography H:
    ├─ If H(keypoint_I) ≈ keypoint_I'
    │  → THIS IS A GOOD KEYPOINT! ✓
    ├─ If NOT matching
    │  → BAD keypoint, discard ✗
    ↓
5️⃣  Create PSEUDO-LABELS:
    "These matched points are TRUE keypoints"
    ↓
6️⃣  Train SuperPoint on these labels
    ├─ Learn to detect at those locations
    ├─ Learn descriptors for those points
    └─ Improve detection quality
    ↓
7️⃣  Next iteration: Use IMPROVED SuperPoint
    └─ It's now better than basic detector!

THIS IS THE LOOP!

# VISUAL EXPLANATION

ITERATION 1 (Start):
─────────────────────
Basic Detector (ORB):        After Homographic Adaptation:
Finds 60 keypoints    ──→    SuperPoint trained
(maybe 30 are good)          Now finds 100 keypoints
                             (60 good ones!)

ITERATION 2:
─────────────────────
SuperPoint (better):         After next iteration:
Finds 100 keypoints   ──→    SuperPoint trained
(60 are good)                Now finds 150 keypoints
                             (90 good ones!)

ITERATION 3:
─────────────────────
SuperPoint (even better):    After next iteration:
Finds 150 keypoints   ──→    SuperPoint trained
(90 are good)                Now finds 200 keypoints
                             (140 good ones!)

... continue for 30 epochs ...

FINAL (After training):
─────────────────────────
SuperPoint (expert):
Finds 300-400 keypoints!
(250+ are good ones!)

# THE KEY INSIGHT

Q: But don't we need features to find features?

A: YES! But we use BOOTSTRAPPING:

   Start with: BASIC detector (50-100 keypoints)
         ↓
   Train SuperPoint on those
         ↓
   SuperPoint is now BETTER
         ↓
   (Optional) Use SP to generate next labels
         ↓
   Train SuperPoint again
         ↓
   SuperPoint is EVEN BETTER
         ↓
   Repeat...
         ↓
   Final: 300-400 keypoints!

IT'S NOT CIRCULAR - IT'S ITERATIVE IMPROVEMENT!

Basic detector → SuperPoint v1 → SuperPoint v2 → ... → Expert
  (50 KPT)       (100 KPT)       (150 KPT)      (300 KPT)

# WHAT HAPPENS IN YOUR OUTDOOR LIDAR CASE

Your scenario: Outdoor featureless terrain

Step 1: Apply basic detector (ORB or FAST)
        Result: 30-50 keypoints in flat field
        └─ These are mostly NOISE or HORIZON edges
        
Step 2: Create 2 warped versions with homography
        ├─ Image 1: Slightly rotated
        ├─ Image 2: Slightly translated
        └─ Detect in both → get 30-50 points each
        
Step 3: Match through homography
        ├─ Which points appear in BOTH?
        ├─ Answer: Very few! (maybe 15-20)
        └─ These 15-20 are the "TRUE" keypoints
        
Step 4: Train SuperPoint on these 15-20 labels
        └─ Learn: "These specific locations matter"
        
Step 5: Next epoch - now SuperPoint is SLIGHTLY BETTER
        ├─ Finds 40-50 keypoints (instead of 30)
        ├─ More of them repeat across frames
        └─ Can be used as new labels
        
Step 6-30: Repeat
        └─ Gradually improve from 30 → 100 → 200 → 300

# SPECIFIC STRATEGY FOR YOUR OUSTER OS1 + OUTDOOR

Data Collection & Preprocessing
────────────────────────────────────────────
✓ Collect 5000-10000 frames from diverse outdoor scenes:
  ├─ Urban (buildings, streets) → More features
  ├─ Rural (fields, trees) → Some features
  ├─ Open areas (parking lots, plains) → Few features
  └─ Include different weather (dry, wet, etc.)

✓ For each frame, extract 4 channels:
  ├─ Range (depth in meters)
  ├─ Reflectivity (material property)
  ├─ Signal (return strength)
  └─ NearIR (laser reflectance)

✓ Preprocess:
  ├─ Normalize each channel separately
  ├─ Handle invalid points (range=0) → set to 0
  ├─ Resize to (256, 256) if needed (optional)
  └─ Save as .npy files

# Homographic Adaptation Training
─────────────────────────────────────────
The KEY INSIGHT:
  Even in featureless terrain, REPEATABLE patterns exist!
  
  When you warp an image, some regions stay similar:
  ├─ Horizon line positions
  ├─ Ground-object boundaries
  ├─ Texture patterns (even subtle)
  ├─ Depth discontinuities
  └─ Intensity variations

  SuperPoint learns to find these via homographic adaptation!

