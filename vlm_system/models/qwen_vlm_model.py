from ..utils.vlm_monitor_window import VLMMonitorWindow
import base64
import io
import time
import re
import numpy as np
import torch as th
from typing import Dict, Any, List, Tuple, Optional
from openai import OpenAI
from PIL import Image
import hashlib
import cv2

from .model_interface import VLMModelInterface
from .model_config import VLMModelConfig


class QwenVLMModel(VLMModelInterface):
    """
    The main program for the interaction between RL and VLM in the EE-RL paper. 
    Please make sure that the API deployment is correct.
    Check the configuration items in model_config.py and train.py
    """
    def __init__(self, config: VLMModelConfig = None, enable_monitor: bool = True):
        if config is None:
            config = VLMModelConfig()
        super().__init__(config)
        
        self.client = None
        self.call_count = 0
        self.error_count = 0
        self.last_call_time = 0
        self.min_call_interval = 1.0
        self.response_times = []
        
        self.reward_cache = {}
        self.cache_hit_count = 0
        self.similar_cache_hit_count = 0
        
        self.image_similarity_threshold = 0.90
        self.state_similarity_threshold = 0.85
        
        self.enable_monitor = enable_monitor
        self.monitor_window = None
        if self.enable_monitor:
            self.monitor_window = VLMMonitorWindow("VLM Inference Monitor")
            self.monitor_window.start()
        
    def load_model(self) -> None:
        try:
            self.client = OpenAI(
                api_key=self.config.api_key,
                base_url=self.config.base_url,
            )
            self.is_loaded = True
        except Exception as e:
            self.is_loaded = False
            raise RuntimeError("Failed to load Qwen VLM model.") from e
    
    def predict_reward(self, input_data: Dict[str, Any]) -> float:
        """Predict reward correction for a single input sample."""
        return self.predict_reward_correction(input_data)
    
    def predict_batch_rewards(self, batch_input: List[Dict[str, Any]]) -> List[float]:
        """Predict reward corrections for a batch with API-rate pacing."""
        rewards = []
        for input_data in batch_input:
            reward = self.predict_reward_correction(input_data)
            rewards.append(reward)
            # Add a short delay to avoid aggressive request bursts.
            time.sleep(0.1)
        return rewards
    
    def predict_reward_correction(self, input_data: Dict[str, Any]) -> float:
        """Run cache-aware VLM inference and return a clipped reward correction."""
        if not self.is_loaded:
            self.load_model()
        
        start_time = time.time()
        
        try:
            # Enforce a minimum interval between external API calls.
            current_time = time.time()
            if current_time - self.last_call_time < self.min_call_interval:
                time.sleep(self.min_call_interval - (current_time - self.last_call_time))
            
            # Extract normalized inference inputs.
            image = input_data.get("image")
            speed = float(input_data.get("speed", 0))
            throttle = float(input_data.get("throttle", 0))
            steer = float(input_data.get("steer", 0))
            current_maneuver = str(input_data.get("current_maneuver", "UNKNOWN"))
            # Parse traffic-light context for prompt construction.
            tl_state_raw = str(input_data.get("traffic_light_state", "Unknown"))
            tl_distance_raw = input_data.get("traffic_light_distance", float('inf'))
            tl_state = tl_state_raw if tl_state_raw else "Unknown"
            if tl_distance_raw == float('inf') or tl_distance_raw is None:
                tl_distance_str = "No traffic light in range"
            else:
                tl_distance_str = f"{float(tl_distance_raw):.1f} m"
            
            # Convert maneuver labels to English for monitor display.
            maneuver_map = {
                "LANEFOLLOW": "Straight",
                "LEFT": "Left Turn",
                "RIGHT": "Right Turn",
                "CHANGELANELEFT": "Lane Change Left",
                "CHANGELANERIGHT": "Lane Change Right",
                "STRAIGHT": "Straight",
                "UNKNOWN": "Unknown",
                "INVALID": "Invalid"
            }
            
            maneuver_cn = maneuver_map.get(current_maneuver.upper(), "Unknown")

            # Update monitor with current frame and driving state.
            if self.monitor_window:
                self.monitor_window.update_image(image)
                self.monitor_window.update_vehicle_status(
                    speed=speed,  # km/h
                    throttle=throttle,
                    steer=steer,
                    maneuver=maneuver_cn
                )
                self.monitor_window.clear_response()
            
            # Build structured state used by cache similarity matching.
            current_state = {
                'speed': speed,
                'throttle': throttle,
                'steer': steer,
                'maneuver': current_maneuver
            }
            
            # Build a deterministic key for optional exact-cache storage.
            cache_key = self._generate_cache_key(image, speed, throttle, steer, current_maneuver)

            """
            ===== Similar-State Cache Lookup =====
            Reuse prior reward corrections when both image and driving
            state pass similarity thresholds.
            """
            similar_cache_result = self._find_similar_cache(image, current_state)
            if similar_cache_result is not None:
                similar_reward, similarity_score = similar_cache_result
                self.cache_hit_count += 1

                # Report cache-hit details in the monitor stream.
                if self.monitor_window:
                    self.monitor_window.update_prompt(
                        f"Similar cache hit (similarity: {similarity_score:.3f}) - Speed: {speed:.1f} km/h, "
                        f"Throttle: {throttle:.3f}, Steer: {steer:.3f}, Maneuver: {maneuver_cn}")
                    self.monitor_window.add_response_chunk(f"[Similar Cache Hit] Reward adjustment: {similar_reward:.4f}")
                    self.monitor_window.finalize_response(similar_reward)
                    self._update_monitor_stats()
                    
                return similar_reward
            
            # Encode image payload for the multimodal API.
            base64_image = self._encode_image(image)
            
            # Build prompts from the active prompt mode.
            active_system_prompt, active_user_prompt_template = self.config.get_active_prompts()
            try:
                user_prompt = active_user_prompt_template.format(
                    speed=speed,
                    throttle=throttle,
                    steer=steer,
                    current_maneuver=f"{maneuver_cn}",
                    tl_state=tl_state,
                    tl_distance_str=tl_distance_str,
                )
            except KeyError as exc:
                raise ValueError(
                    f"Prompt template formatting failed for mode '{self.config.prompt_mode}'. "
                    f"Missing key or unescaped braces: {exc}"
                ) from exc
            
            # Push rendered prompts to monitor for traceability.
            if self.monitor_window:
                full_prompt = f"Prompt Mode: {self.config.prompt_mode}\n\nSystem Prompt: {active_system_prompt}\n\nUser Prompt: {user_prompt}"
                self.monitor_window.update_prompt(full_prompt)
                self.monitor_window.add_response_chunk("Calling VLM API...")
            
            # Build chat-completions request payload.
            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": active_system_prompt}]
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
                        },
                        {"type": "text", "text": user_prompt}
                    ]
                }
            ]
            
            # Execute streaming inference request.
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                stream=True,
            )
            
            response_text = ""
            if self.monitor_window:
                self.monitor_window.add_response_chunk("\n\nVLM Response:\n")
                
            for chunk in completion:
                if chunk.choices[0].delta.content is not None:
                    chunk_content = chunk.choices[0].delta.content
                    response_text += chunk_content
                    
                    if self.monitor_window:
                        self.monitor_window.add_response_chunk(chunk_content)
            
            reward_correction = self._parse_reward_response(response_text)
            
            final_reward = self.postprocess_reward(reward_correction)
            
            self.call_count += 1
            self.last_call_time = time.time()
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            if len(self.response_times) > 100:
                self.response_times.pop(0)
            
            if self.config.enable_cache:
                # Persist structured cache entry for future reuse.
                image_hash = self._compute_image_hash(image)
                self.reward_cache[cache_key] = {
                    'reward': final_reward,
                    'state': current_state,
                    'image_hash': image_hash,
                    'timestamp': current_time
                }
                
                if len(self.reward_cache) > self.config.cache_size:
                    oldest_key = min(self.reward_cache.keys(), 
                                   key=lambda k: self.reward_cache[k]['timestamp'])
                    del self.reward_cache[oldest_key]
            
            if self.monitor_window:
                self.monitor_window.finalize_response(final_reward)
                self._update_monitor_stats()
            
            return final_reward
            
        except Exception as e:
            self.error_count += 1
            
            if self.monitor_window:
                self.monitor_window.add_response_chunk(f"\n\n[Error] {str(e)}")
                self.monitor_window.finalize_response(0.0)
                self._update_monitor_stats()
            
            return 0.0
    
    def _encode_image(self, image: Any) -> str:
        """Encode an input image into a base64 PNG string."""
        if isinstance(image, np.ndarray):
            # Normalize to uint8 before PIL conversion.
            if image.dtype != np.uint8:
                image = (image * 255).astype(np.uint8) if image.max() <= 1.0 else image.astype(np.uint8)
            
            # Convert NumPy array to PIL image.
            pil_image = Image.fromarray(image)
        elif isinstance(image, Image.Image):
            pil_image = image
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
        
        # Resize to configured model input size.
        if pil_image.size != self.config.image_size:
            pil_image = pil_image.resize(self.config.image_size)
        
        # Encode as PNG and return base64 text.
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        buffer.seek(0)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    def _parse_reward_response(self, response_text: str) -> float:
        """Parse a reward value from VLM text response with safe fallbacks."""
        try:
            # Prefer explicit final-value bracket formats.
            bracket_pattern = r'(?:Final\s*reward\s*value|最终奖励值)\s*\[\s*(-?\d*\.?\d+)\s*\]'
            bracket_matches = re.findall(bracket_pattern, response_text, flags=re.IGNORECASE)
            
            if bracket_matches:
                reward = float(bracket_matches[0])
                reward = max(-2, min(2, reward))
                return reward
            
            # Otherwise parse any bracketed number.
            any_bracket_pattern = r'\[\s*(-?\d*\.?\d+)\s*\]'
            any_bracket_matches = re.findall(any_bracket_pattern, response_text)
            
            if any_bracket_matches:
                reward = float(any_bracket_matches[-1])
                reward = max(-2, min(2, reward))
                return reward
            
            # Fall back to generic numeric extraction.
            general_pattern = r'-?\d*\.?\d+'
            general_matches = re.findall(general_pattern, response_text)
            
            if general_matches:
                reward = float(general_matches[0])
                reward = max(-2, min(2, reward))
                return reward
            else:
                if "positive" in response_text.lower() or "good" in response_text.lower():
                    return 2
                elif "negative" in response_text.lower() or "bad" in response_text.lower():
                    return -2
                else:
                    return 0.0
                    
        except Exception:
            return 0.0
    
    def _compute_image_hash(self, image: Any) -> str:
        """Compute a perceptual RGB hash for robust visual cache matching."""
        try:
            # Convert input to a NumPy image array.
            if isinstance(image, np.ndarray):
                img_array = image
            elif isinstance(image, Image.Image):
                img_array = np.array(image)
            else:
                return str(hash(str(image)))
            
            # Ensure RGB channels before resizing and DCT.
            if len(img_array.shape) == 2:
                img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 3:
                if img_array.shape[2] == 4:
                    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
                else:
                    img_rgb = img_array
            else:
                return str(hash(str(image)))
            
            # Use staged downsampling for high-resolution inputs.
            original_height, original_width = img_rgb.shape[:2]
            
            if original_width > 128 or original_height > 128:
                img_intermediate = cv2.resize(img_rgb, (128, 128), interpolation=cv2.INTER_AREA)
                img_resized = cv2.resize(img_intermediate, (64, 64), interpolation=cv2.INTER_AREA)
            else:
                img_resized = cv2.resize(img_rgb, (64, 64), interpolation=cv2.INTER_AREA)

            # Apply light blur to reduce resizing artifacts.
            img_resized = cv2.GaussianBlur(img_resized, (3, 3), 0.5)
            
            # Build per-channel DCT hashes and concatenate them.
            hash_parts = []
            
            for channel in range(3):
                channel_img = img_resized[:, :, channel].astype(np.float32)
                
                dct = cv2.dct(channel_img)
                
                dct_low = dct[:16, :16]
                
                avg = np.mean(dct_low)
                
                channel_hash = ""
                for i in range(16):
                    for j in range(16):
                        channel_hash += "1" if dct_low[i, j] > avg else "0"
                
                hash_parts.append(channel_hash)
            
            combined_hash = f"R{hash_parts[0]}G{hash_parts[1]}B{hash_parts[2]}"
            
            return combined_hash
            
        except Exception:
            # Fall back to lightweight deterministic hashing.
            if isinstance(image, np.ndarray):
                return str(hash(tuple(image.flatten()[::100])))
            else:
                return str(hash(str(image)))
    
    def _compute_image_similarity(self, hash1: str, hash2: str) -> float:
        """Compute similarity between two RGB perceptual hashes in [0, 1]."""
        try:
            def parse_rgb_hash(hash_str):
                if not hash_str.startswith('R') or 'G' not in hash_str or 'B' not in hash_str:
                    return None, None, None
                
                try:
                    r_start = hash_str.find('R') + 1
                    g_start = hash_str.find('G') + 1
                    b_start = hash_str.find('B') + 1
                    
                    r_hash = hash_str[r_start:hash_str.find('G')]
                    g_hash = hash_str[g_start:hash_str.find('B')]
                    b_hash = hash_str[b_start:]
                    
                    expected_length = 256
                    if len(r_hash) != expected_length or len(g_hash) != expected_length or len(b_hash) != expected_length:
                        return None, None, None
                        
                    return r_hash, g_hash, b_hash
                except:
                    return None, None, None
            
            r1, g1, b1 = parse_rgb_hash(hash1)
            r2, g2, b2 = parse_rgb_hash(hash2)
            
            if any(h is None for h in [r1, g1, b1, r2, g2, b2]):
                return self._fallback_similarity(hash1, hash2)
            
            def channel_similarity(h1, h2):
                if len(h1) != len(h2):
                    return 0.0
                diff_count = sum(1 for i in range(len(h1)) if h1[i] != h2[i])
                return 1.0 - (diff_count / len(h1))
            
            r_sim = channel_similarity(r1, r2)
            g_sim = channel_similarity(g1, g2)
            b_sim = channel_similarity(b1, b2)
            
            # Emphasize red/green channels for traffic-light-sensitive scenes.
            overall_similarity = (r_sim * 0.4 + g_sim * 0.4 + b_sim * 0.2)
            
            return overall_similarity
            
        except Exception:
            return self._fallback_similarity(hash1, hash2)

    def _fallback_similarity(self, hash1: str, hash2: str) -> float:
        """Fallback similarity based on normalized Hamming distance."""
        if len(hash1) != len(hash2):
            return 0.0
        
        if len(hash1) == 0:
            return 1.0
            
        diff_count = sum(1 for i in range(len(hash1)) if hash1[i] != hash2[i])
        return 1.0 - (diff_count / len(hash1))
    
    def _compute_state_similarity(self, state1: Dict[str, Any], state2: Dict[str, Any]) -> float:
        """Compute driving-state similarity in [0, 1] for cache retrieval."""
        if state1['maneuver'] != state2['maneuver']:
            return 0.0
        
        speed_diff = abs(state1['speed'] - state2['speed'])
        throttle_diff = abs(state1['throttle'] - state2['throttle'])
        steer_diff = abs(state1['steer'] - state2['steer'])
        
        speed_sim = max(0, 1 - speed_diff / 35.0)
        
        throttle1 = state1['throttle']
        throttle2 = state2['throttle']
        
        if (throttle1 >= 0) != (throttle2 >= 0):
            throttle_sim = max(0, 1 - throttle_diff / 2.0)
            throttle_sim *= 0.5
        else:
            throttle_sim = max(0, 1 - throttle_diff / 2.0)
        
        steer_diff = abs(state1['steer'] - state2['steer'])
        steer_sim = max(0, 1 - steer_diff / 2.0)
        
        overall_similarity = (speed_sim * 0.5 + steer_sim * 0.3 + throttle_sim * 0.2)
        
        return overall_similarity
    
    def _find_similar_cache(self, image: Any, current_state: Dict[str, Any]) -> Optional[Tuple[float, float]]:
        """Find the best cache entry that meets image/state similarity thresholds."""
        if not self.reward_cache:
            return None
        
        current_image_hash = self._compute_image_hash(image)
        best_similarity = 0.0
        best_reward = None
        
        for cache_key, cache_data in self.reward_cache.items():
            image_similarity = self._compute_image_similarity(
                current_image_hash, cache_data['image_hash']
            )
            
            state_similarity = self._compute_state_similarity(
                current_state, cache_data['state']
            )
            
            overall_similarity = (image_similarity * 0.7 + state_similarity * 0.3)
            
            if (overall_similarity > best_similarity and 
                image_similarity >= self.image_similarity_threshold and 
                state_similarity >= self.state_similarity_threshold):
                best_similarity = overall_similarity
                best_reward = cache_data['reward']
        
        if best_reward is not None:
            return best_reward, best_similarity
        else:
            return None

    def _generate_cache_key(self, image: Any, speed: float, 
                           throttle: float, steer: float, current_maneuver: str) -> str:
        """Generate a stable cache key from image hash and discretized controls."""
        try:
            speed_bucket = round(speed * 20) / 20
            throttle_bucket = round(throttle * 50) / 50
            steer_bucket = round(steer * 50) / 50
            maneuver_bucket = current_maneuver
            
            image_hash = self._compute_image_hash(image)
            
            return f"{image_hash}_{speed_bucket}_{throttle_bucket}_{steer_bucket}_{maneuver_bucket}"
        except:
            return f"{time.time()}_{speed}_{throttle}_{steer}_{current_maneuver}"
    
    def get_model_info(self) -> Dict[str, Any]:
        base_info = super().get_model_info()
        total_cache_hits = self.cache_hit_count
        total_calls = self.call_count + total_cache_hits
        
        base_info.update({
            'model_type': 'QwenVLMModel',
            'call_count': self.call_count,
            'error_count': self.error_count,
            'cache_hit_count': self.cache_hit_count,
            'total_cache_hits': total_cache_hits,
            'cache_size': len(self.reward_cache),
            'error_rate': self.error_count / max(1, self.call_count),
            'cache_hit_rate': total_cache_hits / max(1, total_calls),
            'exact_cache_rate': self.cache_hit_count / max(1, total_calls),
        })
        return base_info
   
    def _update_monitor_stats(self):
        if not self.monitor_window:
            return
            
        avg_response_time = sum(self.response_times) / len(self.response_times) if self.response_times else 0
        
        stats = {
            'call_count': self.call_count,
            'error_count': self.error_count,
            'cache_hit_count': self.cache_hit_count,
            'avg_response_time': avg_response_time,
        }
        
        self.monitor_window.update_stats(stats)

    def cleanup(self):
        if self.reward_cache:
            self.reward_cache.clear()
        if self.monitor_window:
            self.monitor_window.stop()
