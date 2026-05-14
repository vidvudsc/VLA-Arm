from __future__ import annotations

from PIL import Image, ImageDraw

from .env import ArmConfig, ArmState, forward_kinematics, object_position


def render_state(state: ArmState, cfg: ArmConfig | None = None) -> Image.Image:
    cfg = cfg or ArmConfig()
    image = Image.new("RGB", (cfg.world_size, cfg.world_size), (12, 17, 24))
    draw = ImageDraw.Draw(image, "RGBA")

    base, joint, end = forward_kinematics(state, cfg)
    reach = cfg.link1 + cfg.link2
    draw.ellipse([base[0] - reach, base[1] - reach, base[0] + reach, base[1] + reach], outline=(255, 255, 255, 20), width=1)
    draw.ellipse(
        [base[0] - cfg.min_spawn_radius, base[1] - cfg.min_spawn_radius, base[0] + cfg.min_spawn_radius, base[1] + cfg.min_spawn_radius],
        outline=(255, 255, 255, 12),
        width=1,
    )

    bx, by = state.bowl.x, state.bowl.y
    br = cfg.bowl_radius
    draw.ellipse([bx - br, by - br, bx + br, by + br], fill=(85, 55, 145, 110), outline=(216, 190, 255, 255), width=4)
    draw.ellipse([bx - 5, by - 5, bx + 5, by + 5], fill=(216, 190, 255, 210))

    ox, oy = object_position(state, cfg)
    obj_color = (245, 86, 66, 255) if not state.placed else (95, 235, 145, 255)
    r = cfg.object_radius
    draw.ellipse([ox - r, oy - r, ox + r, oy + r], fill=obj_color, outline=(255, 244, 232, 255), width=3)

    draw.line([base, joint], fill=(230, 232, 238, 255), width=8)
    draw.line([joint, end], fill=(145, 200, 250, 255), width=8)
    draw.ellipse([base[0] - 13, base[1] - 13, base[0] + 13, base[1] + 13], fill=(80, 86, 102, 255), outline=(245, 245, 245, 255), width=3)
    draw.ellipse([joint[0] - 8, joint[1] - 8, joint[0] + 8, joint[1] + 8], fill=(45, 65, 96, 255), outline=(245, 245, 245, 255), width=2)

    tip_color = (255, 230, 70, 255) if not state.holding else (95, 235, 145, 255)
    draw.ellipse([end[0] - 7, end[1] - 7, end[0] + 7, end[1] + 7], fill=tip_color, outline=(20, 25, 32, 255), width=2)
    if state.holding:
        draw.line([end, (ox, oy)], fill=(95, 235, 145, 175), width=3)

    return image


def image_to_tensor(image: Image.Image):
    import numpy as np
    import torch

    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


def image_to_uint8_tensor(image: Image.Image):
    import numpy as np
    import torch

    arr = np.asarray(image, dtype=np.uint8)
    arr = np.transpose(arr, (2, 0, 1)).copy()
    return torch.from_numpy(arr)
