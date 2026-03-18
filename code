#!/usr/bin/env python3
import argparse
import collections
import logging
import math
import random
import re
import time
import weakref
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

import numpy as np

import pygame
from pygame.locals import (
    KMOD_CTRL, K_ESCAPE, K_q,
    K_UP, K_DOWN, K_LEFT, K_RIGHT,
    K_w, K_s, K_a, K_d,
    K_SPACE, K_h, K_SLASH,
    K_o, K_m, K_v, K_l,
    K_g, K_k, K_t, K_p,
)

try:
    import tkinter as tk
    from tkinter import simpledialog
except Exception:
    tk = None
    simpledialog = None

import carla
from carla import ColorConverter as cc


# =========================
# V2V message definitions
# =========================
class V2VMessageType(Enum):
    HARD_BRAKE_WARNING = auto()
    LANE_MERGE_REQUEST = auto()
    LANE_MERGE_ACCEPT = auto()
    LANE_MERGE_DENY = auto()


@dataclass
class V2VMessage:
    sender_id: int
    message_type: V2VMessageType
    timestamp: float
    position: carla.Location
    velocity: carla.Vector3D
    extra: Optional[dict] = None  # Python 3.8-friendly

    def age(self) -> float:
        return time.time() - self.timestamp


class V2VCommunicationSystem:
    def __init__(self, ego: carla.Vehicle, world: carla.World, hud: "HUD"):
        self.ego = ego
        self.world = world
        self.hud = hud

        self.enabled = True
        self.sent_messages: List[V2VMessage] = []
        self.received_messages: List[V2VMessage] = []

        self.merge_in_progress = False
        self.merge_request_time = 0.0
        self.merge_target_lane = None
        self.merge_responses = {}
        self.stats = {
            "warnings_sent": 0,
            "warnings_received": 0,
            "merge_requests_sent": 0,
            "merge_requests_received": 0,
            "merge_successful": 0,
        }

        # External jamming factor (0 = none, 1 = full jam)
        self.jamming_factor: float = 0.0

    # -------- jamming control --------
    def set_jamming_factor(self, factor: float):
        self.jamming_factor = clamp(factor, 0.0, 1.0)

    # -------- internal helpers --------
    def _get_nearby_vehicles(self, radius=80.0) -> List[carla.Vehicle]:
        ego_loc = self.ego.get_transform().location
        actors = self.world.get_actors().filter("vehicle.*")
        result = []
        for a in actors:
            if a.id == self.ego.id:
                continue
            if ego_loc.distance(a.get_transform().location) <= radius:
                result.append(a)
        return result

    def _broadcast(self, msg: V2VMessage):
        # Simulate packet loss under jamming (higher factor → more loss)
        if self.jamming_factor > 0.0:
            loss_prob = self.jamming_factor
            if random.random() < loss_prob:
                # Message completely lost due to jammer
                return

        self.sent_messages.append(msg)
        if msg.message_type == V2VMessageType.HARD_BRAKE_WARNING:
            self.stats["warnings_sent"] += 1
        elif msg.message_type == V2VMessageType.LANE_MERGE_REQUEST:
            self.stats["merge_requests_sent"] += 1

    # -------- hard brake detection --------
    def check_hard_braking(self, control: carla.VehicleControl):
        if not self.enabled:
            return
        if control.brake > 0.7:
            now = time.time()
            if self.sent_messages and now - self.sent_messages[-1].timestamp < 2.0:
                return
            tf = self.ego.get_transform()
            msg = V2VMessage(
                sender_id=self.ego.id,
                message_type=V2VMessageType.HARD_BRAKE_WARNING,
                timestamp=now,
                position=tf.location,
                velocity=self.ego.get_velocity(),
            )
            self._broadcast(msg)
            self.hud.notification("V2V: Hard brake warning sent", (255, 200, 50), 2.0)

    # -------- merge handling --------
    def request_lane_merge(self, target_lane: int):
        if not self.enabled:
            self.hud.notification("Enable V2V first (press V)", (255, 100, 100), 2.0)
            return
        if self.merge_in_progress:
            self.hud.notification("Merge already in progress", (255, 200, 50), 2.0)
            return
        now = time.time()
        tf = self.ego.get_transform()
        msg = V2VMessage(
            sender_id=self.ego.id,
            message_type=V2VMessageType.LANE_MERGE_REQUEST,
            timestamp=now,
            position=tf.location,
            velocity=self.ego.get_velocity(),
            extra={"target_lane": target_lane},
        )
        self._broadcast(msg)
        self.merge_in_progress = True
        self.merge_request_time = now
        self.merge_target_lane = target_lane
        self.merge_responses = {}
        self.hud.notification("V2V: Lane merge request sent", (100, 255, 100), 3.0)

    def simulate_incoming_warnings(self):
        """For demo: randomly simulate V2V brake warnings from nearby vehicles."""
        if not self.enabled:
            return

        # Under heavy jamming, warnings basically don't arrive
        if self.jamming_factor >= 0.9:
            return

        jam_scale = 1.0 - self.jamming_factor  # 1 → normal, 0 → no messages
        near = self._get_nearby_vehicles()
        now = time.time()
        for v in near:
            if random.random() < 0.002 * jam_scale:
                msg = V2VMessage(
                    sender_id=v.id,
                    message_type=V2VMessageType.HARD_BRAKE_WARNING,
                    timestamp=now,
                    position=v.get_transform().location,
                    velocity=v.get_velocity(),
                )
                self.received_messages.append(msg)
                self.stats["warnings_received"] += 1

    def simulate_merge_responses(self):
        if not self.merge_in_progress:
            return
        now = time.time()
        if now - self.merge_request_time > 6.0:
            self.merge_in_progress = False
            self.hud.notification("⚠️ V2V: Merge timeout", (255, 150, 0), 2.0)
            return

        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location

        for vehicle in self._get_nearby_vehicles():
            if vehicle.id in self.merge_responses:
                continue
            v_loc = vehicle.get_transform().location
            dist = ego_loc.distance(v_loc)
            if dist < 40.0:
                # Under jamming, some responses are simply lost
                if self.jamming_factor > 0.0 and random.random() < self.jamming_factor:
                    continue

                accept = dist > 15.0 or not is_in_front(ego_tf, v_loc, max_angle_deg=45.0)
                response_type = (
                    V2VMessageType.LANE_MERGE_ACCEPT if accept else V2VMessageType.LANE_MERGE_DENY
                )
                msg = V2VMessage(
                    sender_id=vehicle.id,
                    message_type=response_type,
                    timestamp=now,
                    position=v_loc,
                    velocity=vehicle.get_velocity(),
                )
                self.received_messages.append(msg)
                self.merge_responses[vehicle.id] = "accept" if accept else "deny"
                self.stats["merge_requests_received"] += 1

        if len(self.merge_responses) >= 2:
            denies = sum(1 for r in self.merge_responses.values() if r == "deny")
            if denies == 0:
                self.merge_in_progress = False
                self.stats["merge_successful"] += 1
                self.hud.notification(
                    "✓ V2V: Merge approved - safe to proceed", (100, 255, 100), 3.0
                )
            else:
                self.merge_in_progress = False
                self.hud.notification(
                    f"✗ V2V: Merge denied by {denies} vehicle(s)", (255, 100, 100), 3.0
                )

    # -------- main update & query --------
    def update(self):
        if not self.enabled:
            return
        self.simulate_incoming_warnings()
        self.simulate_merge_responses()
        now = time.time()
        self.received_messages = [m for m in self.received_messages if now - m.timestamp < 5.0]
        self.sent_messages = [m for m in self.sent_messages if now - m.timestamp < 5.0]

    def get_active_warnings(self) -> List[V2VMessage]:
        return [
            m
            for m in self.received_messages
            if m.message_type == V2VMessageType.HARD_BRAKE_WARNING and m.age() < 3.0
        ]

    def get_recent_messages(self, max_age: float = 5.0) -> List[V2VMessage]:
        return [m for m in self.received_messages if m.age() < max_age]

    def toggle(self):
        self.enabled = not self.enabled
        self.hud.notification(f"V2V System: {'On' if self.enabled else 'Off'}", (100, 200, 255), 2.0)


# =========================
# Utility helpers
# =========================
def get_actor_display_name(actor, truncate=250):
    if actor is None:
        return "Unknown"
    name = " ".join(actor.type_id.replace("_", ".").title().split(".")[1:])
    return (name[: truncate - 1] + "…") if len(name) > truncate else name


def clamp(n, minn, maxn):
    return max(min(maxn, n), minn)


def is_in_front(ego_transform: carla.Transform, target_location: carla.Location, max_angle_deg=45.0):
    forward = ego_transform.get_forward_vector()
    to_target = target_location - ego_transform.location
    to_target.z = 0.0
    f = np.array([forward.x, forward.y])
    t = np.array([to_target.x, to_target.y])
    if np.linalg.norm(t) < 1e-6:
        return True
    cosang = np.dot(f, t) / (np.linalg.norm(f) * np.linalg.norm(t))
    cosang = float(np.clip(cosang, -1.0, 1.0))
    ang = math.degrees(math.acos(cosang))
    return ang <= max_angle_deg and np.dot(f, t) > 0


# =========================
# Password dialog
# =========================
class PasswordDialog:
    def __init__(self, default_password="1234"):
        self.password = default_password

    def ask(self) -> bool:
        if tk is None or simpledialog is None:
            return True
        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        user_input = simpledialog.askstring(
            "Autopilot Toggle",
            "Enter password to switch mode:",
            show="*",
            parent=root,
        )
        root.destroy()
        if user_input is None:
            return False
        return user_input == self.password


# =========================
# HUD and overlays
# =========================
class FadingText:
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0.0
        self.surface = pygame.Surface(self.dim, flags=pygame.SRCALPHA)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        self.surface.fill((0, 0, 0, 0))
        text_texture = self.font.render(text, True, color)
        self.surface.blit(text_texture, (10, 11))
        self.seconds_left = float(seconds)

    def tick(self, clock):
        if clock is not None:
            delta = 1e-3 * clock.get_time()
            self.seconds_left = max(0.0, self.seconds_left - delta)

    def render(self, display):
        if self.seconds_left > 0.0:
            display.blit(self.surface, self.pos)


class HelpText:
    def __init__(self, font, width, height):
        lines = [
            "CARLA: V2V Communication Demo",
            "WASD/Arrows: drive | SPACE: hand brake | Q: D/R",
            "P: toggle Autopilot (password)",
            "K: toggle AEB | V: toggle V2V System",
            "L: request V2V lane merge",
            "O: toggle perception | M: toggle V2V messages",
            "G: toggle radar | T: switch vehicle type",
            "H or ?: toggle this help | ESC: quit",
            "",
            "V2V Features:",
            "- Hard brake warnings (auto-sent)",
            "- Cooperative lane merge requests",
            "- Real-time message visualization",
            "",
            "Default password: 1234",
        ]
        self.font = font
        self.line_space = 18
        self.dim = (600, len(lines) * self.line_space + 16)
        self.pos = (width / 2 - self.dim[0] / 2, height / 2 - self.dim[1] / 2)
        self.surface = pygame.Surface(self.dim, flags=pygame.SRCALPHA)
        self.surface.fill((0, 0, 0, 170))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (18, n * self.line_space))
        self._render = False

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


class HUD:
    def __init__(self, width, height):
        self.dim = (width, height)
        fonts = [x for x in pygame.font.get_fonts() if "mono" in x] or [
            pygame.font.get_default_font()
        ]
        mono = pygame.font.match_font(fonts[0])
        self._font_small = pygame.font.Font(mono, 14)
        self._font = pygame.font.Font(mono, 18)
        self._font_big = pygame.font.Font(mono, 36)
        self._notifications = FadingText(self._font, (width, 40), (0, height - 42))
        self.help = HelpText(self._font_small, width, height)
        self._show_perception = True
        self._show_v2v_messages = True
        self.server_fps = 0
        self._server_clock = pygame.time.Clock()

        # Perception overlays
        self.prior_warning_active = False
        self.aeb_active_banner = False
        self.predictive_banner = False

        # Radar visualization
        self.radar_points = []
        self.radar_enabled = False

        # V2V system reference (set later)
        self.v2v_system: Optional[V2VCommunicationSystem] = None

        # Speedometer value (km/h)
        self.current_speed_kmh: float = 0.0

        # Jamming visualization state
        self.jamming_active: bool = False
        self.jamming_strength: float = 0.0
        self.jamming_wave_phase: float = 0.0

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        try:
            self.server_fps = self._server_clock.get_fps()
        except Exception:
            pass

    def toggle_perception(self):
        self._show_perception = not self._show_perception

    def toggle_v2v_messages(self):
        self._show_v2v_messages = not self._show_v2v_messages
        status = "Shown" if self._show_v2v_messages else "Hidden"
        self.notification(f"V2V Messages: {status}", (100, 200, 255), 1.5)

    def notification(self, text, color=(255, 255, 255), seconds=2.0):
        self._notifications.set_text(text, color=color, seconds=seconds)

    def set_jamming_state(self, active: bool, strength: float = 0.0):
        self.jamming_active = active
        self.jamming_strength = clamp(strength, 0.0, 1.0) if active else 0.0

    def tick(self, clock):
        self._notifications.tick(clock)
        # Update jamming wave animation
        if self.jamming_active and clock is not None:
            delta = 1e-3 * clock.get_time()
            # Faster phase → waves move outward
            self.jamming_wave_phase = (self.jamming_wave_phase + delta * 2.0) % 10.0

    def render(self, display):
        # FPS
        fps_txt = self._font_small.render(f"FPS: {int(self.server_fps)}", True, (255, 255, 255))
        display.blit(fps_txt, (8, 6))

        # V2V status indicator
        if self.v2v_system:
            v2v_color = (100, 255, 100) if self.v2v_system.enabled else (150, 150, 150)
            v2v_txt = self._font_small.render(
                f'V2V: {"ON" if self.v2v_system.enabled else "OFF"}', True, v2v_color
            )
            display.blit(v2v_txt, (8, 24))

        # SPEEDOMETER (top-right)
        speed_int = int(self.current_speed_kmh)
        label_txt = self._font_small.render("SPEED", True, (255, 255, 255))
        speed_txt = self._font_big.render(f"{speed_int:3d} km/h", True, (255, 255, 255))
        margin = 20
        sx = self.dim[0] - speed_txt.get_width() - margin
        sy = margin
        display.blit(label_txt, (sx, sy))
        display.blit(speed_txt, (sx, sy + 18))

        # Yellow prior warning banner
        if self._show_perception and self.prior_warning_active:
            banner = pygame.Surface((self.dim[0], 44))
            banner.set_alpha(180)
            banner.fill((240, 200, 40))
            txt = self._font.render(
                "⚠ PRIOR COLLISION WARNING: vehicle ahead", True, (0, 0, 0)
            )
            banner.blit(txt, (self.dim[0] // 2 - txt.get_width() // 2, 8))
            display.blit(banner, (0, 52))

        # Blue predictive TTC banner
        if self._show_perception and self.predictive_banner:
            banner = pygame.Surface((self.dim[0], 44))
            banner.set_alpha(180)
            banner.fill((70, 140, 240))
            txt = self._font.render("Predictive AEB Engaged (TTC < 1.5 s)", True, (255, 255, 255))
            banner.blit(txt, (self.dim[0] // 2 - txt.get_width() // 2, 8))
            display.blit(banner, (0, 96))

        # Red AEB banner
        if self._show_perception and self.aeb_active_banner:
            banner = pygame.Surface((self.dim[0], 60))
            banner.set_alpha(180)
            banner.fill((200, 40, 40))
            txt = self._font_big.render("!! EMERGENCY BRAKE ACTIVATED !!", True, (255, 255, 255))
            banner.blit(txt, (self.dim[0] // 2 - txt.get_width() // 2, 8))
            display.blit(banner, (0, 142))

        # V2V Brake Warnings
        if self._show_perception and self.v2v_system and self.v2v_system.enabled:
            warnings = self.v2v_system.get_active_warnings()
            if warnings:
                banner = pygame.Surface((self.dim[0], 50))
                banner.set_alpha(180)
                banner.fill((255, 140, 0))
                txt = self._font.render(
                    f"⚠️ V2V WARNING: {len(warnings)} vehicle(s) braking ahead!",
                    True,
                    (255, 255, 255),
                )
                banner.blit(txt, (self.dim[0] // 2 - txt.get_width() // 2, 12))
                display.blit(banner, (0, 204))

        # V2V Message Feed (bottom-left)
        if self._show_v2v_messages and self.v2v_system and self.v2v_system.enabled:
            messages = self.v2v_system.get_recent_messages(max_age=4.0)
            if messages:
                panel_w, panel_h = 420, 140
                margin = 12
                origin = (margin, self.dim[1] - panel_h - margin)
                panel = pygame.Surface((panel_w, panel_h), flags=pygame.SRCALPHA)
                panel.fill((0, 0, 0, 170))

                title = self._font.render("V2V Messages", True, (255, 255, 255))
                panel.blit(title, (12, 8))
                y_offset = 34
                for m in messages[-4:]:
                    sender = m.sender_id
                    mtype = m.message_type.name.replace("_", " ").title()
                    age = m.age()
                    text = f"{sender}: {mtype} ({age:0.1f}s)"
                    msg_txt = self._font_small.render(text, True, (220, 220, 220))
                    panel.blit(msg_txt, (12, y_offset))
                    y_offset += 22

                stats_y = panel_h - 28
                stats = self.v2v_system.stats
                stats_txt = self._font_small.render(
                    f"Sent: {stats['warnings_sent']} | Recv: {stats['warnings_received']} | "
                    f"Merges: {stats['merge_successful']}",
                    True,
                    (180, 180, 180),
                )
                panel.blit(stats_txt, (8, stats_y))
                display.blit(panel, origin)

        # Radar overlay
        if self.radar_enabled and self.radar_points:
            panel_w, panel_h = 280, 280
            margin = 12
            origin = (self.dim[0] - panel_w - margin, self.dim[1] - panel_h - margin)
            pygame.draw.rect(display, (0, 0, 0), (*origin, panel_w, panel_h), border_radius=8)
            pygame.draw.rect(
                display, (50, 50, 50), (*origin, panel_w, panel_h), width=1, border_radius=8
            )
            cx, cy = origin[0] + panel_w // 2, origin[1] + panel_h // 2
            for r in [40, 80, 120]:
                pygame.draw.circle(display, (60, 60, 60), (cx, cy), r, 1)
            for depth, az, alt, vel in self.radar_points[-256:]:
                r = min(120, depth * 6.0)
                x = cx + r * math.cos(az)
                y = cy + r * math.sin(az)
                size = 2 if vel < 0 else 3
                color = (180, 180, 180) if vel >= 0 else (120, 120, 255)
                pygame.draw.circle(display, color, (int(x), int(y)), size)
            label = self._font_small.render("RADAR", True, (200, 200, 200))
            display.blit(label, (origin[0] + 8, origin[1] + 6))

        # Jamming attack visualization (red wave from bottom-center)
        if self.jamming_active:
            num_rings = 4
            base_radius = 30
            max_radius = 140
            cx = self.dim[0] // 2
            cy = self.dim[1] - 120

            for i in range(num_rings):
                t = (self.jamming_wave_phase + i * 0.8) % 3.0
                radius = int(base_radius + (max_radius - base_radius) * (t / 3.0))
                alpha = int(255 * (1.0 - t / 3.0) * (0.4 + 0.6 * self.jamming_strength))
                if alpha <= 0 or radius <= 0:
                    continue
                surf = pygame.Surface((radius * 2, radius * 2), flags=pygame.SRCALPHA)
                color = (255, 60, 60, max(40, alpha // 2))
                pygame.draw.circle(surf, color, (radius, radius), radius, width=3)
                display.blit(surf, (cx - radius, cy - radius))

            jam_txt = self._font.render("JAMMER VEHICLE ATTACK", True, (255, 120, 120))
            display.blit(
                jam_txt,
                (cx - jam_txt.get_width() // 2, cy - max_radius - 28),
            )

        self._notifications.render(display)
        self.help.render(display)


# =========================
# Sensors
# =========================
class CollisionSensor:
    def __init__(self, parent_actor, hud: HUD):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find("sensor.other.collision")
        self.sensor = world.try_spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        if self.sensor:
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for f, intensity in self.history:
            history[f] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        other = event.other_actor
        self.hud.notification(
            "Collision with %r" % get_actor_display_name(other), (255, 0, 0), seconds=3.0
        )
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


class LaneInvasionSensor:
    def __init__(self, parent_actor, hud: HUD):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find("sensor.other.lane_invasion")
        self.sensor = world.try_spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        if self.sensor:
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ["%r" % str(x).split()[-1] for x in lane_types]
        self.hud.notification("Crossed line %s" % " and ".join(text))


class GnssSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self.lat = 0.0
        self.lon = 0.0
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find("sensor.other.gnss")
        self.sensor = world.try_spawn_actor(
            bp,
            carla.Transform(carla.Location(x=1.0, z=2.8)),
            attach_to=parent_actor,
        )
        if self.sensor:
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: GnssSensor._on_gnss(weak_self, event))

    @staticmethod
    def _on_gnss(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


class IMUSensor:
    def __init__(self, parent_actor):
        self.sensor = None
        self.accelerometer = (0.0, 0.0, 0.0)
        self.gyroscope = (0.0, 0.0, 0.0)
        self.compass = 0.0
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find("sensor.other.imu")
        self.sensor = world.try_spawn_actor(bp, carla.Transform(), attach_to=parent_actor)
        if self.sensor:
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda event: IMUSensor._on_imu(weak_self, event))

    @staticmethod
    def _on_imu(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.accelerometer = (
            event.accelerometer.x,
            event.accelerometer.y,
            event.accelerometer.z,
        )
        self.gyroscope = (event.gyroscope.x, event.gyroscope.y, event.gyroscope.z)
        self.compass = event.compass


class CameraManager:
    def __init__(self, parent_actor, hud: HUD, gamma):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        world = parent_actor.get_world()
        bound_x = 0.5 + parent_actor.bounding_box.extent.x
        bound_z = 0.5 + parent_actor.bounding_box.extent.z
        cam_transform = carla.Transform(
            carla.Location(x=-4.0 * bound_x, z=2.0 * bound_z),
            carla.Rotation(pitch=8.0),
        )
        bp = world.get_blueprint_library().find("sensor.camera.rgb")
        bp.set_attribute("image_size_x", str(hud.dim[0]))
        bp.set_attribute("image_size_y", str(hud.dim[1]))
        bp.set_attribute("gamma", str(gamma))
        self.sensor = world.try_spawn_actor(bp, cam_transform, attach_to=parent_actor)
        if self.sensor:
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        image.convert(cc.Raw)
        arr = np.frombuffer(image.raw_data, dtype=np.uint8)
        arr = np.reshape(arr, (image.height, image.width, 4))
        arr = arr[:, :, :3][:, :, ::-1]
        self.surface = pygame.surfarray.make_surface(arr.swapaxes(0, 1))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))


class RadarSensor:
    def __init__(self, parent_actor, hud: HUD, range_m=50.0):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        self.enabled = True
        world = parent_actor.get_world()
        bp = world.get_blueprint_library().find("sensor.other.radar")
        bp.set_attribute("horizontal_fov", "50")
        bp.set_attribute("vertical_fov", "25")
        bp.set_attribute("range", str(float(range_m)))
        transform = carla.Transform(carla.Location(x=2.0, z=1.0))
        self.sensor = world.try_spawn_actor(bp, transform, attach_to=parent_actor)
        if self.sensor:
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda data: RadarSensor._on_radar(weak_self, data))
        self.hud.radar_enabled = True

    def destroy(self):
        try:
            if self.sensor:
                self.sensor.stop()
                self.sensor.destroy()
        except Exception:
            pass
        self.sensor = None
        self.hud.radar_points = []
        self.hud.radar_enabled = False

    @staticmethod
    def _on_radar(weak_self, radar_data):
        self = weak_self()
        if not self:
            return
        pts = []
        for d in radar_data:
            pts.append((d.depth, d.azimuth, d.altitude, d.velocity))
        self.hud.radar_points = pts[-512:]


# =========================
# Collision Predictor + AEB (speed-dependent)
# =========================
class PerceptionAndAEB:
    def __init__(self, ego_vehicle: carla.Vehicle, hud: HUD):
        self.ego = ego_vehicle
        self.hud = hud

        # Base distances for low speeds
        self.prior_min = 10.0   # meters
        self.prior_max = 15.0
        self.aeb_min = 7.0
        self.aeb_max = 11.0

        self.aeb_enabled = True
        self.aeb_active = False
        self.aeb_release_time = 0.0
        self.aeb_cooldown_until = 0.0

        self._predictive_engaged = False

    def toggle_aeb(self):
        self.aeb_enabled = not self.aeb_enabled
        self.hud.notification(
            f"Collision-Avoidance (AEB): {'On' if self.aeb_enabled else 'Off'}",
            seconds=1.5,
        )

    def _nearest_ahead(self):
        world = self.ego.get_world()
        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_vel = self.ego.get_velocity()
        ego_speed = math.sqrt(ego_vel.x**2 + ego_vel.y**2 + ego_vel.z**2)

        nearest_d = None
        nearest_actor = None
        rel_speed = 0.0

        for actor in world.get_actors():
            if actor.id == self.ego.id:
                continue
            if not (
                actor.type_id.startswith("vehicle.")
                or actor.type_id.startswith("walker.pedestrian")
            ):
                continue
            loc = actor.get_transform().location
            if not is_in_front(ego_tf, loc, max_angle_deg=45.0):
                continue
            d = ego_loc.distance(loc)
            if nearest_d is None or d < nearest_d:
                nearest_d = d
                nearest_actor = actor
                avel = actor.get_velocity()
                a_speed = math.sqrt(avel.x**2 + avel.y**2 + avel.z**2)
                rel_speed = max(0.0, ego_speed - a_speed)

        return nearest_d, nearest_actor, rel_speed

    def step(self, dt_sec: float):
        now = time.time()

        # Reset HUD flags
        self.hud.prior_warning_active = False
        self.hud.aeb_active_banner = False
        self.hud.predictive_banner = False
        self._predictive_engaged = False

        if not self.aeb_enabled:
            return

        # Ego speed in km/h
        vel = self.ego.get_velocity()
        ego_speed_mps = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
        ego_speed_kmh = ego_speed_mps * 3.6

        # Speed-dependent scaling: 1.0 at <=30, 1.5 at >=60
        base_prior_min = self.prior_min
        base_prior_max = self.prior_max
        base_aeb_min = self.aeb_min
        base_aeb_max = self.aeb_max

        if ego_speed_kmh <= 30.0:
            speed_factor = 1.0
        elif ego_speed_kmh >= 60.0:
            speed_factor = 1.5
        else:
            speed_factor = 1.0 + 0.5 * ((ego_speed_kmh - 30.0) / 30.0)

        prior_min = base_prior_min * speed_factor
        prior_max = base_prior_max * speed_factor
        aeb_min = base_aeb_min * speed_factor
        aeb_max = base_aeb_max * speed_factor

        # Nearest object ahead
        dist, actor, rel_speed = self._nearest_ahead()

        # Prior visual warning
        if dist is not None and prior_min <= dist <= prior_max:
            self.hud.prior_warning_active = True

        # If AEB currently active → keep full brake for a few seconds
        if self.aeb_active:
            self.hud.aeb_active_banner = True
            ctrl = self.ego.get_control()
            ctrl.throttle = 0.0
            ctrl.brake = 1.0
            ctrl.hand_brake = False
            self.ego.apply_control(ctrl)
            if now >= self.aeb_release_time:
                self.aeb_active = False
                self.hud.aeb_active_banner = False
                self.aeb_cooldown_until = now + 10.0
                self.hud.notification(
                    "AEB released; re-armed in 10s", (255, 255, 0), seconds=2.0
                )
            return

        if dist is None:
            return

        # TTC-based trigger
        ttc_triggered = False
        if rel_speed > 0.1:
            ttc = dist / rel_speed
            if ttc < 1.5 and now >= self.aeb_cooldown_until:
                ttc_triggered = True
                self._predictive_engaged = True
                self.hud.predictive_banner = True

        # Distance-based trigger (speed-dependent)
        in_distance_window = (
            aeb_min <= dist <= aeb_max and now >= self.aeb_cooldown_until
        )

        if ttc_triggered or in_distance_window:
            soft_brake_start = aeb_min * 1.03  # slightly above aeb_min
            if dist > soft_brake_start and not ttc_triggered:
                num = (aeb_max - dist)
                den = max(1e-3, (aeb_max - aeb_min))
                brake_strength = float(np.clip(num / den, 0.4, 1.0))

                ctrl = self.ego.get_control()
                ctrl.throttle = 0.0
                ctrl.brake = brake_strength
                ctrl.hand_brake = False
                self.ego.apply_control(ctrl)
                self.hud.predictive_banner = True
                return

            # Hard AEB stop
            self.aeb_active = True
            self.aeb_release_time = now + 3.0
            self.hud.aeb_active_banner = True
            msg = "Emergency Brake Activated (3s)"
            if ttc_triggered:
                msg = "Predictive AEB Stop (3s)"
            self.hud.notification(msg, (255, 80, 80), seconds=2.0)

            ctrl = self.ego.get_control()
            ctrl.throttle = 0.0
            ctrl.brake = 1.0
            ctrl.hand_brake = False
            self.ego.apply_control(ctrl)


# =========================
# Jammer Vehicle Attack Manager
# =========================
class JammerAttackManager:
    """
    Spawns a malicious jammer vehicle and simulates a jamming attack:
    - Red jamming wave animation on HUD
    - V2V messages dropped probabilistically
    - Ego vehicle slows down safely
    """

    def __init__(
        self,
        carla_world: carla.World,
        ego: carla.Vehicle,
        hud: HUD,
        v2v_system: Optional[V2VCommunicationSystem],
    ):
        self.world = carla_world
        self.ego = ego
        self.hud = hud
        self.v2v_system = v2v_system

        self.jammer_actor: Optional[carla.Vehicle] = None
        self.jam_radius: float = 80.0
        self.max_factor: float = 0.95
        self.active: bool = False

        self._spawn_jammer()

    def _spawn_jammer(self):
        try:
            lib = self.world.get_blueprint_library()
            v_bps = lib.filter("vehicle.*")
            if not v_bps:
                return
            bp = random.choice(v_bps)
            if bp.has_attribute("color"):
                # Try to pick a red-like color if available
                reds = [
                    c
                    for c in bp.get_attribute("color").recommended_values
                    if "255,0,0" in c or "200,0,0" in c
                ]
                if reds:
                    bp.set_attribute("color", random.choice(reds))
                else:
                    bp.set_attribute(
                        "color",
                        random.choice(bp.get_attribute("color").recommended_values),
                    )
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                return
            ego_loc = self.ego.get_transform().location
            random.shuffle(spawn_points)
            sp = None
            for cand in spawn_points:
                if ego_loc.distance(cand.location) > 30.0:
                    sp = cand
                    break
            if sp is None:
                sp = spawn_points[0]
            self.jammer_actor = self.world.try_spawn_actor(bp, sp)
            if self.jammer_actor:
                # Let jammer drive like a normal NPC
                try:
                    self.jammer_actor.set_autopilot(True)
                except Exception:
                    pass
                self.hud.notification(
                    "Malicious jammer vehicle spawned", (255, 120, 120), 3.0
                )
        except Exception as e:
            logging.warning("Failed to spawn jammer vehicle: %s", e)

    def destroy(self):
        if self.jammer_actor:
            try:
                self.jammer_actor.destroy()
            except Exception:
                pass
            self.jammer_actor = None
        if self.v2v_system:
            self.v2v_system.set_jamming_factor(0.0)
        self.hud.set_jamming_state(False, 0.0)

    def tick(self, dt_sec: float):
        if not self.jammer_actor:
            self.active = False
            if self.v2v_system:
                self.v2v_system.set_jamming_factor(0.0)
            self.hud.set_jamming_state(False, 0.0)
            return

        ego_loc = self.ego.get_transform().location
        jam_loc = self.jammer_actor.get_transform().location
        dist = ego_loc.distance(jam_loc)

        if dist <= self.jam_radius:
            # Map distance → jamming factor (near: strong, far: weak)
            factor = self.max_factor * max(0.0, 1.0 - dist / self.jam_radius)
            self.active = True

            # Tell V2V system about jamming
            if self.v2v_system:
                self.v2v_system.set_jamming_factor(factor)

            # Update HUD (animation + state)
            self.hud.set_jamming_state(True, factor)

            # Strong jamming → show warning text
            if factor > 0.7:
                self.hud.notification(
                    "⚠ Jamming Detected – Switching Frequency & Slowing Down",
                    (255, 120, 120),
                    seconds=2.0,
                )

            # Slow ego down safely (soft override)
            try:
                ctrl = self.ego.get_control()
                # Cap throttle
                ctrl.throttle = min(ctrl.throttle, 0.35)
                # Mild brake when jamming is strong
                if factor > 0.5:
                    ctrl.brake = max(ctrl.brake, 0.2)
                ctrl.hand_brake = False
                self.ego.apply_control(ctrl)
            except Exception:
                pass
        else:
            # Out of range → no jamming
            self.active = False
            if self.v2v_system:
                self.v2v_system.set_jamming_factor(0.0)
            self.hud.set_jamming_state(False, 0.0)


# =========================
# World wrapper
# =========================
class World:
    VEHICLE_SEQUENCE = ["car", "bike", "truck"]

    def __init__(self, carla_world: carla.World, hud: HUD, args):
        self.world = carla_world
        self.map = self.world.get_map()
        self.hud = hud
        self.args = args

        self.player: Optional[carla.Actor] = None
        self.collision_sensor = None
        self.lane_invasion_sensor = None
        self.gnss_sensor = None
        self.imu_sensor = None
        self.camera_manager = None
        self.radar_sensor: Optional[RadarSensor] = None

        self._gamma = args.gamma

        self.vehicle_kind = "car"
        self._traffic_manager = None
        self.npc_actors = []

        self.perception_aeb: Optional[PerceptionAndAEB] = None
        self.v2v_system: Optional[V2VCommunicationSystem] = None
        self.jammer_manager: Optional[JammerAttackManager] = None

        self._spawn_player(kind=self.vehicle_kind)
        self._spawn_sensors()
        self._spawn_npcs(num=20)

        # Create jammer attack system after ego + sensors + NPCs are ready
        self.jammer_manager = JammerAttackManager(
            self.world, self.player, self.hud, self.v2v_system
        )

        self.world.on_tick(self.hud.on_world_tick)

    def get_traffic_manager(self):
        return self._traffic_manager

    def _select_blueprint_for_kind(self, kind: str):
        lib = self.world.get_blueprint_library()
        v_bps = lib.filter("vehicle.*")

        def find_by_contains(substrs, fallback=None):
            cands = [bp for bp in v_bps if any(s in bp.id for s in substrs)]
            return random.choice(cands) if cands else fallback

        if kind == "bike":
            bike = find_by_contains(
                ["bike", "bicycle", "bh.crossbike", "diamondback.century"]
            )
            if not bike:
                two = [
                    bp
                    for bp in v_bps
                    if bp.has_attribute("number_of_wheels")
                    and bp.get_attribute("number_of_wheels").as_int() == 2
                ]
                if two:
                    bike = random.choice(two)
            return bike or random.choice(v_bps)

        if kind == "truck":
            truck = find_by_contains(
                ["truck", "pickup", "carlacola", "t2", "van", "volkswagen.t2", "fordcrown"]
            )
            return truck or random.choice(v_bps)

        car = find_by_contains(
            [
                "tesla.model3",
                "audi.tt",
                "dodge",
                "nissan",
                "mini",
                "mercedes",
                "lincoln",
                "seat",
                "mustang",
            ]
        )
        return car or random.choice(v_bps)

    def _spawn_player(self, kind="car"):
        bp = self._select_blueprint_for_kind(kind)
        bp.set_attribute("role_name", "hero")
        if bp.has_attribute("color"):
            bp.set_attribute(
                "color", random.choice(bp.get_attribute("color").recommended_values)
            )
        spawn_points = self.map.get_spawn_points() if self.map else [carla.Transform()]
        spawn_point = random.choice(spawn_points)
        self.player = self.world.try_spawn_actor(bp, spawn_point)
        while self.player is None:
            spawn_point = random.choice(spawn_points)
            self.player = self.world.try_spawn_actor(bp, spawn_point)
        self.hud.notification(f"Spawned ego: {get_actor_display_name(self.player)}")

    def _respawn_player_same_pose(self, next_kind):
        tf = self.player.get_transform()
        vel = self.player.get_velocity()
        try:
            self.player.destroy()
        except Exception:
            pass
        bp = self._select_blueprint_for_kind(next_kind)
        bp.set_attribute("role_name", "hero")
        if bp.has_attribute("color"):
            bp.set_attribute(
                "color", random.choice(bp.get_attribute("color").recommended_values)
            )
        self.player = self.world.try_spawn_actor(bp, tf)
        tries = 0
        while self.player is None and tries < 20:
            tf.location.x += 0.2
            tries += 1
            self.player = self.world.try_spawn_actor(bp, tf)
        if self.player:
            self.player.set_target_velocity(vel)
            self.hud.notification(
                f"Vehicle switched: {next_kind.upper()} → {get_actor_display_name(self.player)}"
            )

    def _spawn_sensors(self):
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.imu_sensor = IMUSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud, self._gamma)
        if self.radar_sensor:
            self.radar_sensor.destroy()
        self.radar_sensor = RadarSensor(self.player, self.hud, range_m=50.0)

        self.perception_aeb = PerceptionAndAEB(self.player, self.hud)
        self.v2v_system = V2VCommunicationSystem(self.player, self.world, self.hud)
        self.hud.v2v_system = self.v2v_system

    def _destroy_sensors(self):
        try:
            if self.camera_manager and self.camera_manager.sensor:
                self.camera_manager.sensor.destroy()
        except Exception:
            pass
        for s in [
            self.collision_sensor,
            self.lane_invasion_sensor,
            self.gnss_sensor,
            self.imu_sensor,
        ]:
            try:
                if s and getattr(s, "sensor", None):
                    s.sensor.stop()
                    s.sensor.destroy()
            except Exception:
                pass
        if self.radar_sensor:
            self.radar_sensor.destroy()
            self.radar_sensor = None

    def _spawn_npcs(self, num=20):
        # Create / reuse Traffic Manager
        self._traffic_manager = (
            self._traffic_manager
            or carla.Client(self.args.host, self.args.port).get_trafficmanager()
        )
        tm = self._traffic_manager
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.global_percentage_speed_difference(10.0)  # NPCs ~10% slower

        blueprints = self.world.get_blueprint_library().filter("vehicle.*")
        spawn_points = self.map.get_spawn_points()
        random.shuffle(spawn_points)
        count = 0
        for sp in spawn_points:
            if count >= num:
                break
            bp = random.choice(blueprints)
            if bp.has_attribute("color"):
                bp.set_attribute(
                    "color", random.choice(bp.get_attribute("color").recommended_values)
                )
            npc = self.world.try_spawn_actor(bp, sp)
            if npc:
                npc.set_autopilot(True, tm.get_port())
                self.npc_actors.append(npc)
                count += 1
        self.hud.notification(
            f"Spawned NPC vehicles: {count}", (160, 220, 255), 2.0
        )

    def destroy_npcs(self):
        for a in self.npc_actors:
            try:
                a.destroy()
            except Exception:
                pass
        self.npc_actors = []
        if self.jammer_manager:
            self.jammer_manager.destroy()
            self.jammer_manager = None

    def tick(self, dt_sec):
        # Update ego speed for HUD speedometer
        if self.player is not None:
            vel = self.player.get_velocity()
            speed_mps = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)
            self.hud.current_speed_kmh = speed_mps * 3.6  # m/s → km/h

        if self.perception_aeb:
            self.perception_aeb.step(dt_sec)
        if self.v2v_system:
            self.v2v_system.update()
        if self.jammer_manager:
            self.jammer_manager.tick(dt_sec)

    def render(self, display):
        if self.camera_manager:
            self.camera_manager.render(display)
        self.hud.render(display)

    def cycle_vehicle(self):
        idx = World.VEHICLE_SEQUENCE.index(self.vehicle_kind)
        next_kind = World.VEHICLE_SEQUENCE[(idx + 1) % len(World.VEHICLE_SEQUENCE)]
        self.vehicle_kind = next_kind
        self._destroy_sensors()
        self._respawn_player_same_pose(next_kind)
        self._spawn_sensors()

    def toggle_radar(self):
        if self.radar_sensor and self.radar_sensor.sensor:
            self.radar_sensor.destroy()
            self.hud.notification("Radar: OFF", (255, 255, 0), 1.5)
        else:
            self.radar_sensor = RadarSensor(self.player, self.hud, range_m=50.0)
            self.hud.notification("Radar: ON", (0, 255, 180), 1.5)


# =========================
# Keyboard controller
# =========================
class KeyboardControl:
    def __init__(self, world: World, tm: carla.TrafficManager, start_in_autopilot=False):
        self.world = world
        self.tm = tm
        self._autopilot_enabled = start_in_autopilot
        self._control = carla.VehicleControl()
        self._lights = carla.VehicleLightState.NONE
        self._steer_cache = 0.0
        self.pwd = PasswordDialog()

        # Configure ego vehicle for TrafficManager-based autopilot
        if isinstance(world.player, carla.Vehicle):
            world.player.set_autopilot(self._autopilot_enabled, self.tm.get_port())

            # Make ego a bit faster and allow auto lane change
            try:
                self.tm.auto_lane_change(world.player, True)
                self.tm.vehicle_percentage_speed_difference(world.player, -20.0)  # 20% faster
                self.tm.ignore_lights_percentage(world.player, 0)
            except Exception:
                pass

            world.player.set_light_state(self._lights)

        if self.world.perception_aeb:
            self.world.perception_aeb.aeb_enabled = not self._autopilot_enabled

        self.world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def parse_events(self, clock):
        ret_quit = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                ret_quit = True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    ret_quit = True
                elif event.key in (K_h, K_SLASH):
                    self.world.hud.help.toggle()
                elif event.key == K_o:
                    self.world.hud.toggle_perception()
                elif event.key == K_m:
                    self.world.hud.toggle_v2v_messages()
                elif event.key == K_v:
                    if self.world.v2v_system:
                        self.world.v2v_system.toggle()
                elif event.key == K_l:
                    if self.world.v2v_system:
                        self.world.v2v_system.request_lane_merge(target_lane=1)
                    else:
                        self.world.hud.notification(
                            "Enable V2V first (press V)", (255, 100, 100), 2.0
                        )
                elif event.key == K_g:
                    self.world.toggle_radar()
                elif event.key == K_k:
                    self.world.perception_aeb.toggle_aeb()
                elif event.key == K_t:
                    self.world.cycle_vehicle()
                elif event.key == K_q:
                    self._control.gear = -1 if self._control.gear >= 0 else 1
                    self.world.hud.notification(
                        f"Gear: {'Reverse' if self._control.gear < 0 else 'Drive'}",
                        (180, 255, 180),
                        1.3,
                    )
                elif event.key == K_p:
                    # Toggle autopilot with password
                    if self.pwd.ask():
                        self._autopilot_enabled = not self._autopilot_enabled
                        try:
                            self.world.player.set_autopilot(
                                self._autopilot_enabled, self.tm.get_port()
                            )
                            # Ensure lane change remains allowed
                            self.tm.auto_lane_change(self.world.player, True)
                            self.tm.vehicle_percentage_speed_difference(
                                self.world.player, -20.0
                            )
                        except Exception:
                            pass

                        if self.world.perception_aeb:
                            self.world.perception_aeb.aeb_enabled = (
                                not self._autopilot_enabled
                            )
                        self.world.hud.notification(
                            f"Autopilot: {'On' if self._autopilot_enabled else 'Off'}"
                        )
                    else:
                        self.world.hud.notification(
                            "Incorrect password", (255, 80, 80), 2.0
                        )
                elif event.key == K_c:
                    try:
                        presets = [
                            x
                            for x in dir(carla.WeatherParameters)
                            if re.match("[A-Z].+", x)
                        ]
                        preset = getattr(
                            carla.WeatherParameters, random.choice(presets)
                        )
                        self.world.world.set_weather(preset)
                        self.world.hud.notification("Weather changed")
                    except Exception:
                        pass

        # Manual control only if not in autopilot and AEB not fully active
        if not self._autopilot_enabled and not self.world.perception_aeb.aeb_active:
            keys = pygame.key.get_pressed()
            self._parse_vehicle_keys(keys, clock.get_time())
            self._control.reverse = self._control.gear < 0

            if self.world.v2v_system:
                self.world.v2v_system.check_hard_braking(self._control)

            current_lights = self._lights
            if self._control.brake:
                current_lights |= carla.VehicleLightState.Brake
            else:
                current_lights &= ~carla.VehicleLightState.Brake
            if self._control.reverse:
                current_lights |= carla.VehicleLightState.Reverse
            else:
                current_lights &= ~carla.VehicleLightState.Reverse
            if current_lights != self._lights:
                self._lights = current_lights
                try:
                    self.world.player.set_light_state(
                        carla.VehicleLightState(self._lights)
                    )
                except Exception:
                    pass

            self.world.player.apply_control(self._control)

        return ret_quit

    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.02, 1.00)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.25, 1.0)
        else:
            self._control.brake = 0.0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0.0
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0.0
            self._steer_cache += steer_increment
        else:
            self._steer_cache *= 0.8

        self._steer_cache = clamp(self._steer_cache, -0.7, 0.7)
        self._control.steer = round(self._steer_cache, 3)
        self._control.hand_brake = keys[K_SPACE]

    @staticmethod
    def _is_quit_shortcut(key):
        return key == K_ESCAPE or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# =========================
# Main loop (robust)
# =========================
def game_loop(args):
    pygame.init()
    pygame.font.init()
    world_wrapper = None
    orig_settings = None
    sim_world = None
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(20.0)
        sim_world = client.get_world()

        if args.sync:
            orig_settings = sim_world.get_settings()
            settings = sim_world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            sim_world.apply_settings(settings)
            try:
                tm = client.get_trafficmanager()
                tm.set_synchronous_mode(True)
            except Exception:
                logging.warning("Could not set TrafficManager to sync mode")

        display = pygame.display.set_mode(
            (args.width, args.height), pygame.HWSURFACE | pygame.DOUBLEBUF
        )
        pygame.display.set_caption("CARLA - V2V + Speedometer + Speed-Based AEB + Jammer Attack")

        hud = HUD(args.width, args.height)
        world_wrapper = World(sim_world, hud, args)

        # Use the same Traffic Manager the world used for NPCs
        tm = world_wrapper.get_traffic_manager()
        if tm is None:
            tm = client.get_trafficmanager()

        controller = KeyboardControl(world_wrapper, tm, args.autopilot)

        clock = pygame.time.Clock()
        while True:
            # Advance simulation
            try:
                if args.sync:
                    sim_world.tick()
                else:
                    sim_world.wait_for_tick()
            except Exception as e:
                logging.warning("World tick failed: %s", e)
                continue

            dt = clock.tick_busy_loop(60) / 1000.0

            # Update world logic
            try:
                world_wrapper.tick(dt)
            except Exception as e:
                logging.exception("Error in world tick: %s", e)

            # Input events
            try:
                if controller.parse_events(clock):
                    return
            except Exception as e:
                logging.exception("Error in event handling: %s", e)

            # HUD tick (for animations)
            try:
                hud.tick(clock)
            except Exception as e:
                logging.exception("Error in HUD tick: %s", e)

            # Render
            try:
                world_wrapper.render(display)
            except Exception as e:
                logging.exception("Error in render: %s", e)

            pygame.display.flip()

    finally:
        try:
            if sim_world is not None and orig_settings:
                sim_world.apply_settings(orig_settings)
        except Exception:
            pass
        if world_wrapper:
            try:
                world_wrapper.destroy_npcs()
            except Exception:
                pass
            try:
                world_wrapper._destroy_sensors()
            except Exception:
                pass
            try:
                if world_wrapper.player:
                    world_wrapper.player.destroy()
            except Exception:
                pass
        pygame.quit()


def main():
    parser = argparse.ArgumentParser(
        description="CARLA V2V Demo with Speedometer, Speed-Based AEB & Jammer Attack"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, default=2000)
    parser.add_argument(
        "-a", "--autopilot", action="store_true", help="start with autopilot enabled"
    )
    parser.add_argument(
        "--res", default="1280x720", help="window resolution WxH (default 1280x720)"
    )
    parser.add_argument("--gamma", type=float, default=2.2)
    parser.add_argument("--sync", action="store_true", help="enable synchronous mode")
    args = parser.parse_args()

    try:
        args.width, args.height = [int(x) for x in args.res.split("x")]
    except Exception:
        args.width, args.height = 1280, 720

    logging.basicConfig(
        format="%(levelname)s: %(message)s", level=logging.INFO
    )
    logging.info("Connecting to %s:%s", args.host, args.port)

    try:
        game_loop(args)
    except KeyboardInterrupt:
        print("\nCancelled by user.")


if __name__ == "__main__":
    main()
