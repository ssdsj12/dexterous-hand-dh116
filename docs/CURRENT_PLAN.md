# DH116 褰撳墠鎵ц璁″垝锛?026-11-16 鏇存柊锛?
- **浠跨湡骞冲彴**锛欼saac Sim 4.5 + Isaac Lab 0.54.4 + CUDA Cosserat Rod (`ropesim`, `damping=0.0`, `drag=0.2`)锛屽叏 GPU 闆舵嫹璐濊€﹀悎锛坄set_mesh_poses_all` / `mesh_wrenches_view` / `positions`锛夈€?- **浼犳劅鍣ㄦ柟妗?*锛氫繚鎸佸浐瀹氬墠鏂瑰弻鐩浉鏈猴紙涓嶄娇鐢ㄨ厱閮ㄨ瑙夛級銆?- **缁熶竴鎺ュ彛**锛氭暀甯堜笌瀛︾敓鍏辩敤鍚屼竴濂?24D 鍔ㄤ綔鎺ュ彛锛坄tabletop_surface_contract.py`锛夈€?- **鎺ㄨ繘璺嚎**锛?  1. 鍗曠幆澧冪函 GPU 闆舵嫹璐濊窇閫?鈮?.5 s锝?0 s 闀垮洖鍚堟崗鍙栦笌鎶崌锛屾嬁鍒伴娆℃煍鎬х嚎鎶撳彇鎴愬姛锛?  2. 鍚戦噺鍖栧鐜锛坄num_envs = 16 -> 64`锛夛紝浣跨敤 `flexible_train_curriculum_stage1.json` 璇剧▼鍖栨垚鍔熼棬妲涗笌骞虫粦濂栧姳璁粌鐗规潈鏁欏笀锛堝彲鐢ㄨ剼鏈?MPC 绀鸿寖鍋?BC 鐑惎鍔?+ PPO锛夛紱
  3. 閲囬泦鍚堟牸鏁欏笀绀鸿寖锛岀粨鍚堝浐瀹氬墠鏂瑰弻鐩娴嬭繘琛?BC + DAgger 瀛︾敓钂搁銆?EOF

