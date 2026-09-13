# CourtVision AI (AthletaTrack)

> Physics-informed multi-object tracking for broadcast football that resolves identity switching in dense scrums, paired with an automated FotMob-style physical and spatial analytics engine.

---

## Overview

In broadcast sports video, standard multi-object trackers (ByteTrack, DeepSORT, Kalman filters) fail during high-density contact events—such as set-piece walls, corner kick scrums, and sliding tackles. When players overlap ($\text{IoU} > 0.6$), visual Re-ID features blend together and linear motion assumptions break down, resulting in severe **identity switches (IDSw)**.

**CourtVision AI** introduces a **Kinematic Memory Module** that monitors bounding-box collision graphs. When a scramble occurs, it mutes corrupted visual embeddings, caches pre-collision momentum vectors, and projects dynamic "ghost trajectories" through the huddle. Upon cluster separation, identities are re-assigned via physics-gated cost matching and upper-body kit clustering.

The resulting continuous coordinate stream is mapped via **4-point pitch homography** into metric pitch space ($105\text{m} \times 68\text{m}$) to generate broadcast-grade analytics: **2D Gaussian KDE heatmaps, sprint counts ($>25\text{ km/h}$), distance covered, and dynamic match ratings**.

---

## Architecture Pipeline

```text
[Raw Broadcast Match Video (1080p)]
                │
                ▼
  [YOLOv11 Detection Engine] ──────────► Players, Referees, Goalkeepers, Ball
                │
        ┌───────┴───────┐
        ▼               ▼
[Kit Color Cluster] [4-Point Homography]
(HSV K-Means)       (Pixel u,v ──► Pitch X,Y meters)
        │               │
        └───────┬───────┘
                ▼
      [Base Tracker: ByteTrack]
                │
       [IoU > 0.60 Cluster?]
        ├── No  ──► Standard Bipartite Matching
        └── Yes ──► [Kinematic Memory Module]
                    ├── Cache pre-collision momentum (Vx, Vy, Ax, Ay)
                    ├── Mute visual Re-ID weights
                    ├── Extrapolate ghost trajectories
                    └── Physics-gated Hungarian matching upon separation
                                │
                                ▼
         [Clean Continuous Trajectory Stream: (ID, Frame, X_m, Y_m)]
                                │
       ┌────────────────────────┼────────────────────────┐
       ▼                        ▼                        ▼
[Physical Engine]       [Spatial Heatmaps]      [FotMob Match Ratings]
• Total Distance (km)   • 2D Gaussian KDE       • Base 6.0 + impact score
• Sprints (>25 km/h)    • Zone occupancy        • High-intensity outputs
• Top Speed (km/h)      • Positional drift      • Possession contribution
                                │
                                ▼
                       [FastAPI REST API] ──► Webhooks / Mobile App / Replay JSON
