## 21. Progress Update — Static Target Geolocation

Following the earlier investigation of the static off-axis projection mismatch, the static geolocation pipeline was refined and validated end-to-end.

Current static setup:
- Stationary target at approximately (2, 5, 0.01) m
- UAV hovering at approximately 10 m
- Gimbal fixed around RC7 = 1400
- Target GPS is used only as ground truth for validation, not as an estimator input

### 21.1 Pixel measurement investigation

Two image measurements were compared:
- Bounding-box center
- Contour centroid

Using the same geometric transformation, a 100-sample A/B test produced:

- Bounding-box center: horizontal RMSE ≈ 8.18 cm
- Contour centroid: horizontal RMSE ≈ 4.66 cm

The contour centroid was therefore selected as the current pixel measurement.

A visual projection test also showed that the measured contour centroid and the theoretical forward projection were typically separated by less than 1 pixel.

### 21.2 Realtime state acquisition

The earlier realtime implementation repeatedly queried `gz model` for UAV, gimbal, and pitch-link states. This caused intermittent state acquisition failures and stale state ages of several seconds.

A persistent `/world/hexacopter_runway/dynamic_pose/info` stream was therefore implemented.

The stream provides the required UAV, gimbal, and pitch-link poses continuously and is used to reconstruct the camera pose through the local hierarchy.

The updated realtime implementation no longer relies on repeated `gz model` queries.

### 21.3 Final static validation

A final 100-sample end-to-end validation was performed using:
Camera → target detection → contour centroid → camera ray → full camera/gimbal/UAV transformation → NED → target-plane intersection → UAV GPS → estimated target GPS.

Final result:

- Horizontal RMSE = 4.12 cm
- Horizontal MAE = 4.11 cm
- Maximum horizontal error = 5.32 cm
- North bias = +2.59 cm
- East bias = +3.01 cm

State freshness during validation:
- Mean pose age = 8.9 ms
- Maximum pose age = 78.4 ms
- Mean GPS age = 154.2 ms
- Maximum GPS age = 233.9 ms

The static geometric geolocation stage is considered sufficiently validated for the current simulation baseline. No further geometry tuning is planned at this stage unless a later experiment exposes a significant issue.

Next development stage: EKF-based target state estimation.

## 21. New Files — Static Geolocation Progress

The following files were created during the latest static-target geolocation development and validation:

- `pixel_projection_overlay.py`
  → melihat centroid vs theoretical projection

- `live_pixel_ab_validation.py`
  → membandingkan bbox center vs contour centroid

- `gazebo_state_stream_test.py`
  → validasi `dynamic_pose/info` sebagai sumber state kontinu

- `static_geolocation_realtime.py`
  → realtime geolocation dengan persistent state stream

- `final_static_validation.py`
  → validasi final 100 sampel end-to-end

- `final_static_validation.csv`
  → data mentah 100 sampel hasil validasi final
