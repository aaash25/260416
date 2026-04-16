import taichi as ti
import numpy as np

ti.init(arch=ti.vulkan if ti.vulkan else ti.cpu)

RES = 800
EPS = 1e-4
INF = 1e10

pixels = ti.Vector.field(3, dtype=ti.f32, shape=(RES, RES))

# 场景参数
cam_pos = ti.Vector([0.0, 0.0, 5.0])
light_pos = ti.Vector([2.0, 3.0, 4.0])
bg_color = ti.Vector([0.02, 0.1, 0.15]) 

# 几何体定义
sphere_center = ti.Vector([-1.2, -0.2, 0.0])
sphere_radius = 1.2
cone_v = ti.Vector([1.2, 1.2, 0.0]) 
cone_dir = ti.Vector([0.0, -1.0, 0.0]).normalized() 
cone_angle = 0.5 

@ti.func
def intersect_sphere(o, d):
    co = o - sphere_center
    a = d.dot(d)
    b = 2.0 * co.dot(d)
    c = co.dot(co) - sphere_radius**2
    det = b*b - 4.0*a*c
    t = INF
    if det > 0:
        t1 = (-b - ti.sqrt(det)) / (2.0*a)
        if t1 > EPS: t = t1
    return t

@ti.func
def intersect_cone(o, d):
    cos_a = ti.cos(cone_angle)
    v = o - cone_v
    dv = d.dot(cone_dir)
    vv = v.dot(cone_dir)
    a = dv**2 - cos_a**2
    b = 2.0 * (dv * vv - v.dot(d) * cos_a**2)
    c = vv**2 - v.dot(v) * cos_a**2
    det = b*b - 4.0*a*c
    t = INF
    # 处理 a = 0 的情况，避免除以零错误
    if a != 0:
        if det > 0:
            t1 = (-b - ti.sqrt(det)) / (2.0*a)
            t2 = (-b + ti.sqrt(det)) / (2.0*a)
            # 修复点：改用显式判断代替 Python list 循环
            # 修复圆锥体高度限制，使其与圆锥体顶点位置关联
            if t1 > EPS:
                p1 = o + t1 * d
                if p1.y <= cone_v.y and p1.y >= cone_v.y - 2.6:  # 圆锥体高度为 2.6
                    t = t1
            if t2 > EPS:
                p2 = o + t2 * d
                if p2.y <= cone_v.y and p2.y >= cone_v.y - 2.6 and t2 < t:  # 圆锥体高度为 2.6
                    t = t2
        elif det == 0:
            # 处理光线与圆锥体相切的情况
            t1 = -b / (2.0*a)
            if t1 > EPS:
                p1 = o + t1 * d
                if p1.y <= cone_v.y and p1.y >= cone_v.y - 2.6:  # 圆锥体高度为 2.6
                    t = t1
    return t

@ti.func
def get_normal(p, hit_type):
    n = ti.Vector([0.0, 0.0, 0.0])
    if hit_type == 1: 
        n = (p - sphere_center).normalized()
    else: 
        cp = p - cone_v
        n = (cp - (cp.dot(cone_dir) / ti.cos(cone_angle)**2) * cone_dir).normalized()
    return n

@ti.kernel
def render(ka: ti.f32, kd: ti.f32, ks: ti.f32, shininess: ti.f32):
    for i, j in pixels:
        uv = ti.Vector([(i - 0.5 * RES) / RES, (j - 0.5 * RES) / RES])
        d = ti.Vector([uv.x, uv.y, -1.0]).normalized() 
        
        t_s = intersect_sphere(cam_pos, d)
        t_c = intersect_cone(cam_pos, d)
        
        t_min, hit_type = INF, 0
        if t_s < t_c: t_min, hit_type = t_s, 1
        else: t_min, hit_type = t_c, 2
            
        if t_min < INF:
            hit_p = cam_pos + t_min * d
            n = get_normal(hit_p, hit_type)
            l = (light_pos - hit_p).normalized()
            v = (cam_pos - hit_p).normalized()
            h = (l + v).normalized() 
            
            shadow_hit = False
            s_o = hit_p + n * EPS
            if hit_type == 1:
                if intersect_cone(s_o, l) < (light_pos - hit_p).norm(): shadow_hit = True
            else:
                if intersect_sphere(s_o, l) < (light_pos - hit_p).norm(): shadow_hit = True
            
            base_col = ti.Vector([0.8, 0.1, 0.1]) if hit_type == 1 else ti.Vector([0.6, 0.2, 0.8])
            color = ka * base_col 
            if not shadow_hit:
                color += kd * base_col * max(0.0, n.dot(l))
                color += ks * ti.Vector([1.0, 1.0, 1.0]) * pow(max(0.0, n.dot(h)), shininess)
            pixels[i, j] = color
        else:
            pixels[i, j] = bg_color

window = ti.ui.Window("Vulkan Ray Caster 2026", (RES, RES))
canvas = window.get_canvas()
gui = window.get_gui()

ka, kd, ks, shininess = 0.2, 0.7, 0.5, 32.0

while window.running:
    render(ka, kd, ks, shininess)
    canvas.set_image(pixels)
    with gui.sub_window("Shader Settings", 0.05, 0.05, 0.3, 0.2):
        ka = gui.slider_float("Ka", ka, 0.0, 1.0)
        kd = gui.slider_float("Kd", kd, 0.0, 1.0)
        ks = gui.slider_float("Ks", ks, 0.0, 1.0)
        shininess = gui.slider_float("Shininess", shininess, 1.0, 128.0)
    window.show()
