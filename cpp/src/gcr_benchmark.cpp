#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace gcr {

constexpr double EPS = 1e-9;
constexpr double PI = 3.141592653589793238462643383279502884;

struct Vec2 {
  double x = 0.0;
  double y = 0.0;
};

using Cable = std::vector<Vec2>;
using Polygon = std::vector<Vec2>;

struct Circle {
  Vec2 center;
  double radius = 0.0;
};

struct Workspace {
  double xmin = 0.0;
  double xmax = 1.0;
  double ymin = 0.0;
  double ymax = 0.6;
};

struct ObstacleSet {
  std::vector<Polygon> polygons;
  std::vector<Circle> circles;
};

struct Scenario {
  std::string name;
  std::string title;
  std::string source;
  Workspace workspace;
  ObstacleSet obstacles;
  Cable start;
  Cable goal;
  double epsilon = 0.22;
  int seed = 1;
  std::vector<Vec2> guide_points;
  std::vector<Cable> guide_configs;
};

struct PlannerParams {
  int max_iterations = 1600;
  double scenario_time_limit_s = 120.0;
  bool has_time_limit = true;
  int apf_steps = 8;
  double eta = 0.035;
  double k_att = 0.22;
  double beta = 0.0055;
  double sigma = 0.060;
  int segment_samples = 3;
  int transition_samples = 6;
  double epsilon = 0.115;
  double ell_min_ratio = 0.65;
  double ell_max_ratio = 1.45;
  double dsafe = 0.014;
  double delta_l = 0.075;
  int relax_iterations = 24;
  double relax_step = 0.060;
  double lambda_s = 0.34;
  double lambda_b = 0.075;
  double tracking_weight = 1.0;
  double alpha_def = 0.10;
  double max_apf_node_step = 0.006;
  int max_parent_checks = 28;
  bool shortcut_refinement = true;
  bool endpoint_reorder = false;
  bool robot_execution_feasibility = false;
  double robot_tcp_clearance = 0.0;
  bool has_robot_obstacle_clearance = false;
  double robot_obstacle_clearance = 0.0;
  double robot_swept_tcp_clearance = 0.0;
  bool preserve_terminal_order = true;
  double endpoint_reorder_cost_ratio = 0.75;
  double endpoint_reorder_tangent_margin = 0.05;
  bool robot_base_aware_endpoint_reorder = false;
  bool robot_base_vector_endpoint_reorder = false;
  bool robot_tcp_vector_sector_reorder = false;
  bool robot_tcp_vector_sector_feasibility = false;
  double robot_base_vector_max_turn_rad = 100.0 * PI / 180.0;
  double robot_base_vector_margin_rad = 5.0 * PI / 180.0;
  double robot_tcp_vector_sector_margin_rad = 5.0 * PI / 180.0;
  double max_cable_axis_turn_rad = PI;
  double robot_base_arm_clearance = 0.0;
  int robot_base_transition_samples = 8;
  int shortcut_passes = 3;
  bool has_shortcut_max_distance_factor = false;
  double shortcut_max_distance_factor = 0.0;
  int shortcut_validation_samples = 100;
  double goal_bias = 0.06;
  double tree_bias = 0.84;
  double broad_bias = 0.10;
  int exact_clearance_period = 4;
};

struct TreeNode {
  Cable config;
  int parent = -1;
  double cost = 0.0;
};

struct SearchTree {
  std::string name;
  std::vector<TreeNode> nodes;
};

struct Metrics {
  std::string scenario;
  bool success = false;
  bool timed_out = false;
  int iterations = 0;
  double time_s = 0.0;
  double time_limit_s = std::numeric_limits<double>::quiet_NaN();
  int start_vertices = 0;
  int goal_vertices = 0;
  int raw_path_configs = 0;
  int refined_path_configs = 0;
  bool shortcut_refinement_enabled = true;
  bool endpoint_reorder_enabled = false;
  bool robot_execution_feasibility_enabled = false;
  double robot_tcp_clearance_m = 0.0;
  double robot_obstacle_clearance_m = 0.0;
  double robot_swept_tcp_clearance_m = 0.0;
  bool preserve_terminal_order_enabled = true;
  int endpoint_order_reversals = 0;
  double epsilon_m = 0.0;
  double max_edge_distance_m = std::numeric_limits<double>::quiet_NaN();
  int shortcut_validation_samples = 0;
  double path_cost_J = std::numeric_limits<double>::quiet_NaN();
  double raw_path_cost_J = std::numeric_limits<double>::quiet_NaN();
  double shortcut_reduction_pct = 0.0;
  double deformation_energy = std::numeric_limits<double>::quiet_NaN();
  double min_clearance_m = std::numeric_limits<double>::quiet_NaN();
  double max_segment_length_error_m = std::numeric_limits<double>::quiet_NaN();
  bool static_collision_free = false;
  bool transition_path_free = false;
  int reject_static = 0;
  int reject_no_parent = 0;
  int reject_no_connect = 0;
};

struct PlanResult {
  Scenario scenario;
  bool success = false;
  bool timed_out = false;
  std::vector<Cable> raw_path;
  std::vector<Cable> path;
  int iterations = 0;
  double elapsed_s = 0.0;
  SearchTree tree_start;
  SearchTree tree_goal;
  Metrics metrics;
};

Vec2 operator+(const Vec2& a, const Vec2& b) { return {a.x + b.x, a.y + b.y}; }
Vec2 operator-(const Vec2& a, const Vec2& b) { return {a.x - b.x, a.y - b.y}; }
Vec2 operator*(const Vec2& a, double s) { return {a.x * s, a.y * s}; }
Vec2 operator*(double s, const Vec2& a) { return a * s; }
Vec2 operator/(const Vec2& a, double s) { return {a.x / s, a.y / s}; }
Vec2& operator+=(Vec2& a, const Vec2& b) {
  a.x += b.x;
  a.y += b.y;
  return a;
}
Vec2& operator-=(Vec2& a, const Vec2& b) {
  a.x -= b.x;
  a.y -= b.y;
  return a;
}

double dot(const Vec2& a, const Vec2& b) { return a.x * b.x + a.y * b.y; }
double norm2(const Vec2& a) { return dot(a, a); }
double norm(const Vec2& a) { return std::sqrt(norm2(a)); }
double distance(const Vec2& a, const Vec2& b) { return norm(a - b); }
Vec2 normalized_or(const Vec2& v, const Vec2& fallback = {1.0, 0.0}) {
  const double n = norm(v);
  return n < EPS ? fallback : v / n;
}

double deg2rad(double degrees) { return degrees * PI / 180.0; }
double wrap_angle(double angle) {
  while (angle <= -PI) angle += 2.0 * PI;
  while (angle > PI) angle -= 2.0 * PI;
  return angle;
}
double angle_of(const Vec2& v) { return std::atan2(v.y, v.x); }
double angle_distance(double a, double b) { return std::abs(wrap_angle(a - b)); }

Polygon rectangle(double xmin, double ymin, double xmax, double ymax) {
  return {{xmin, ymin}, {xmax, ymin}, {xmax, ymax}, {xmin, ymax}};
}

Cable line_nodes(const Vec2& start, const Vec2& end, int n_nodes) {
  Cable out;
  out.reserve(static_cast<std::size_t>(n_nodes));
  for (int i = 0; i < n_nodes; ++i) {
    const double t = n_nodes == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n_nodes - 1);
    out.push_back((1.0 - t) * start + t * end);
  }
  return out;
}

Cable polyline_nodes(const std::vector<Vec2>& points, int n_nodes) {
  std::vector<double> cumulative(points.size(), 0.0);
  for (std::size_t i = 1; i < points.size(); ++i) {
    cumulative[i] = cumulative[i - 1] + distance(points[i], points[i - 1]);
  }
  const double total = cumulative.empty() ? 0.0 : cumulative.back();
  if (total < EPS) {
    return Cable(static_cast<std::size_t>(n_nodes), points.empty() ? Vec2{} : points.front());
  }
  Cable out;
  out.reserve(static_cast<std::size_t>(n_nodes));
  std::size_t j = 0;
  for (int i = 0; i < n_nodes; ++i) {
    const double s = total * static_cast<double>(i) / static_cast<double>(n_nodes - 1);
    while (j + 1 < cumulative.size() - 1 && cumulative[j + 1] < s) {
      ++j;
    }
    const double span = cumulative[j + 1] - cumulative[j];
    const double alpha = span < EPS ? 0.0 : (s - cumulative[j]) / span;
    out.push_back((1.0 - alpha) * points[j] + alpha * points[j + 1]);
  }
  return out;
}

Cable circular_arc_nodes(const Vec2& center, double radius, double start_angle, double end_angle, int n_nodes) {
  Cable out;
  out.reserve(static_cast<std::size_t>(n_nodes));
  for (int i = 0; i < n_nodes; ++i) {
    const double t = n_nodes == 1 ? 0.0 : static_cast<double>(i) / static_cast<double>(n_nodes - 1);
    const double a = (1.0 - t) * start_angle + t * end_angle;
    out.push_back({center.x + radius * std::cos(a), center.y + radius * std::sin(a)});
  }
  return out;
}

Cable straight_at(const Vec2& center, double angle_rad, int n_nodes, double segment_length = 0.030) {
  const double length = static_cast<double>(n_nodes - 1) * segment_length;
  const Vec2 d{std::cos(angle_rad), std::sin(angle_rad)};
  return line_nodes(center - 0.5 * length * d, center + 0.5 * length * d, n_nodes);
}

Cable arc_at(const Vec2& center, double theta_start, double theta_end, int n_nodes, double chord = 0.030) {
  const double dtheta = (theta_end - theta_start) / static_cast<double>(n_nodes - 1);
  const double radius = chord / (2.0 * std::sin(std::abs(dtheta) / 2.0));
  return circular_arc_nodes(center, radius, theta_start, theta_end, n_nodes);
}

Cable rounded_l_down_right(const Vec2& top, int n_nodes, double chord = 0.030, int vertical_segments = 3,
                           int arc_segments = 3) {
  const int n_segments = n_nodes - 1;
  arc_segments = std::min(std::max(2, arc_segments), n_segments - 2);
  vertical_segments = std::min(std::max(1, vertical_segments), n_segments - arc_segments - 1);
  const int horizontal_segments = n_segments - vertical_segments - arc_segments;

  Cable points;
  points.reserve(static_cast<std::size_t>(n_nodes));
  points.push_back(top);
  for (int i = 1; i <= vertical_segments; ++i) {
    points.push_back({top.x, top.y - chord * static_cast<double>(i)});
  }

  const double delta = 0.5 * PI / static_cast<double>(arc_segments);
  const double radius = chord / (2.0 * std::sin(0.5 * delta));
  const Vec2 arc_start = points.back();
  const Vec2 arc_center{arc_start.x + radius, arc_start.y};
  for (int i = 1; i <= arc_segments; ++i) {
    const double theta = PI + static_cast<double>(i) * delta;
    points.push_back({arc_center.x + radius * std::cos(theta), arc_center.y + radius * std::sin(theta)});
  }

  const Vec2 bend_end = points.back();
  for (int i = 1; i <= horizontal_segments; ++i) {
    points.push_back({bend_end.x + chord * static_cast<double>(i), bend_end.y});
  }
  return points;
}

std::vector<double> segment_lengths(const Cable& cable) {
  std::vector<double> lengths;
  if (cable.size() < 2) return lengths;
  lengths.reserve(cable.size() - 1);
  for (std::size_t i = 0; i + 1 < cable.size(); ++i) {
    lengths.push_back(distance(cable[i + 1], cable[i]));
  }
  return lengths;
}

double cable_length(const Cable& cable) {
  const auto lengths = segment_lengths(cable);
  return std::accumulate(lengths.begin(), lengths.end(), 0.0);
}

double nominal_segment_length(const Cable& cable) {
  const auto lengths = segment_lengths(cable);
  if (lengths.empty()) return 0.0;
  return std::accumulate(lengths.begin(), lengths.end(), 0.0) / static_cast<double>(lengths.size());
}

double configuration_distance(const Cable& a, const Cable& b) {
  double best_sq = 0.0;
  for (std::size_t i = 0; i < a.size(); ++i) {
    best_sq = std::max(best_sq, norm2(a[i] - b[i]));
  }
  return std::sqrt(best_sq);
}

double deformation_energy(const Cable& next, const Cable& parent) {
  double e = 0.0;
  for (std::size_t i = 0; i + 1 < next.size(); ++i) {
    const Vec2 seg_q = (next[i + 1] - next[i]) - (parent[i + 1] - parent[i]);
    e += norm2(seg_q);
  }
  for (std::size_t i = 0; i + 2 < next.size(); ++i) {
    const Vec2 bend_q = (next[i + 2] - 2.0 * next[i + 1] + next[i]) -
                        (parent[i + 2] - 2.0 * parent[i + 1] + parent[i]);
    e += norm2(bend_q);
  }
  return e;
}

double point_to_segment_distance(const Vec2& point, const Vec2& start, const Vec2& end) {
  const Vec2 seg = end - start;
  const double len_sq = norm2(seg);
  if (len_sq < EPS) return distance(point, start);
  const double t = std::clamp(dot(point - start, seg) / len_sq, 0.0, 1.0);
  return distance(point, start + t * seg);
}

Vec2 closest_point_on_segment(const Vec2& point, const Vec2& start, const Vec2& end) {
  const Vec2 seg = end - start;
  const double len_sq = norm2(seg);
  if (len_sq < EPS) return start;
  const double t = std::clamp(dot(point - start, seg) / len_sq, 0.0, 1.0);
  return start + t * seg;
}

int orientation(const Vec2& a, const Vec2& b, const Vec2& c) {
  const double value = (b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x);
  if (std::abs(value) < EPS) return 0;
  return value > 0.0 ? 1 : -1;
}

bool on_segment(const Vec2& a, const Vec2& b, const Vec2& c) {
  return std::min(a.x, c.x) - EPS <= b.x && b.x <= std::max(a.x, c.x) + EPS &&
         std::min(a.y, c.y) - EPS <= b.y && b.y <= std::max(a.y, c.y) + EPS &&
         orientation(a, b, c) == 0;
}

bool segments_intersect(const Vec2& a, const Vec2& b, const Vec2& c, const Vec2& d) {
  const int o1 = orientation(a, b, c);
  const int o2 = orientation(a, b, d);
  const int o3 = orientation(c, d, a);
  const int o4 = orientation(c, d, b);
  if (o1 != o2 && o3 != o4) return true;
  return (o1 == 0 && on_segment(a, c, b)) || (o2 == 0 && on_segment(a, d, b)) ||
         (o3 == 0 && on_segment(c, a, d)) || (o4 == 0 && on_segment(c, b, d));
}

double segment_to_segment_distance(const Vec2& a, const Vec2& b, const Vec2& c, const Vec2& d) {
  if (segments_intersect(a, b, c, d)) return 0.0;
  return std::min({point_to_segment_distance(a, c, d),
                   point_to_segment_distance(b, c, d),
                   point_to_segment_distance(c, a, b),
                   point_to_segment_distance(d, a, b)});
}

bool point_in_polygon_core(const Vec2& point, const Polygon& polygon) {
  bool inside = false;
  std::size_t j = polygon.size() - 1;
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    const Vec2& pi = polygon[i];
    const Vec2& pj = polygon[j];
    if (on_segment(pj, point, pi)) return true;
    const bool yi_gt = pi.y > point.y;
    const bool yj_gt = pj.y > point.y;
    if (yi_gt != yj_gt) {
      const double x_cross = (pj.x - pi.x) * (point.y - pi.y) / (pj.y - pi.y + EPS) + pi.x;
      if (point.x < x_cross) inside = !inside;
    }
    j = i;
  }
  return inside;
}

bool point_in_polygon(const Vec2& point, const Polygon& polygon) {
  if (polygon.empty()) return false;
  double min_x = polygon[0].x, max_x = polygon[0].x;
  double min_y = polygon[0].y, max_y = polygon[0].y;
  for (const auto& p : polygon) {
    min_x = std::min(min_x, p.x);
    max_x = std::max(max_x, p.x);
    min_y = std::min(min_y, p.y);
    max_y = std::max(max_y, p.y);
  }
  if (point.x < min_x - EPS || point.x > max_x + EPS || point.y < min_y - EPS || point.y > max_y + EPS) {
    return false;
  }
  return point_in_polygon_core(point, polygon);
}

std::pair<Vec2, double> closest_point_on_polygon(const Vec2& point, const Polygon& polygon) {
  Vec2 best{};
  double best_d = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    const Vec2 candidate = closest_point_on_segment(point, polygon[i], polygon[(i + 1) % polygon.size()]);
    const double d = distance(point, candidate);
    if (d < best_d) {
      best_d = d;
      best = candidate;
    }
  }
  return {best, best_d};
}

bool segment_intersects_polygon(const Vec2& start, const Vec2& end, const Polygon& polygon) {
  if (point_in_polygon(start, polygon) || point_in_polygon(end, polygon)) return true;
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    if (segments_intersect(start, end, polygon[i], polygon[(i + 1) % polygon.size()])) return true;
  }
  return false;
}

bool polygon_intersects_polygon(const Polygon& a, const Polygon& b) {
  for (std::size_t i = 0; i < a.size(); ++i) {
    for (std::size_t j = 0; j < b.size(); ++j) {
      if (segments_intersect(a[i], a[(i + 1) % a.size()], b[j], b[(j + 1) % b.size()])) return true;
    }
  }
  for (const auto& p : a) {
    if (point_in_polygon(p, b)) return true;
  }
  for (const auto& p : b) {
    if (point_in_polygon(p, a)) return true;
  }
  return false;
}

bool polygon_intersects_circle(const Polygon& poly, const Vec2& center, double radius) {
  if (point_in_polygon(center, poly)) return true;
  for (const auto& p : poly) {
    if (distance(p, center) <= radius + EPS) return true;
  }
  for (std::size_t i = 0; i < poly.size(); ++i) {
    if (point_to_segment_distance(center, poly[i], poly[(i + 1) % poly.size()]) <= radius + EPS) return true;
  }
  return false;
}

bool segment_intersects_circle(const Vec2& start, const Vec2& end, const Vec2& center, double radius) {
  return point_to_segment_distance(center, start, end) <= radius + EPS;
}

double segment_clearance_to_polygon(const Vec2& start, const Vec2& end, const Polygon& polygon) {
  if (segment_intersects_polygon(start, end, polygon)) return 0.0;
  double best = std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < polygon.size(); ++i) {
    best = std::min(best, segment_to_segment_distance(start, end, polygon[i], polygon[(i + 1) % polygon.size()]));
  }
  return best;
}

double segment_clearance_to_obstacles(const Vec2& start, const Vec2& end, const ObstacleSet& obstacles) {
  double best = std::numeric_limits<double>::infinity();
  for (const auto& poly : obstacles.polygons) {
    best = std::min(best, segment_clearance_to_polygon(start, end, poly));
  }
  for (const auto& circle : obstacles.circles) {
    best = std::min(best, point_to_segment_distance(circle.center, start, end) - circle.radius);
  }
  return best;
}

bool segment_clearance_free(const Vec2& start, const Vec2& end, const ObstacleSet& obstacles, double clearance) {
  if (clearance <= EPS) return segment_clearance_to_obstacles(start, end, obstacles) > EPS;
  return segment_clearance_to_obstacles(start, end, obstacles) >= clearance - EPS;
}

struct SamplePoint {
  int k = 0;
  double rho = 0.0;
  Vec2 point;
};

std::vector<SamplePoint> segment_sample_points(const Cable& cable, int samples_per_segment) {
  std::vector<SamplePoint> samples;
  const int denom = std::max(samples_per_segment, 1);
  for (std::size_t k = 0; k + 1 < cable.size(); ++k) {
    for (int q = 0; q <= samples_per_segment; ++q) {
      const double rho = static_cast<double>(q) / static_cast<double>(denom);
      samples.push_back({static_cast<int>(k), rho, (1.0 - rho) * cable[k] + rho * cable[k + 1]});
    }
  }
  return samples;
}

Cable clip_to_workspace(const Cable& cable, const Workspace& workspace, double margin = 0.010) {
  Cable out = cable;
  for (auto& p : out) {
    p.x = std::clamp(p.x, workspace.xmin + margin, workspace.xmax - margin);
    p.y = std::clamp(p.y, workspace.ymin + margin, workspace.ymax - margin);
  }
  return out;
}

Cable project_uniform_segment_lengths(const Cable& cable, double desired_length, int n_sweeps = 10) {
  Cable result = cable;
  if (result.size() < 2) return result;
  for (int sweep = 0; sweep < n_sweeps; ++sweep) {
    for (int k = 0; k < static_cast<int>(result.size()) - 1; ++k) {
      Vec2 d = result[k + 1] - result[k];
      const double len = norm(d);
      if (len < EPS) continue;
      const Vec2 corr = d * (0.5 * (len - desired_length) / len);
      result[k] += corr;
      result[k + 1] -= corr;
    }
    for (int k = static_cast<int>(result.size()) - 2; k >= 0; --k) {
      Vec2 d = result[k + 1] - result[k];
      const double len = norm(d);
      if (len < EPS) continue;
      const Vec2 corr = d * (0.5 * (len - desired_length) / len);
      result[k] += corr;
      result[k + 1] -= corr;
    }
  }
  return result;
}

Cable project_segment_vector_deviation(const Cable& cable, const Cable& reference, double delta_l,
                                       bool fixed_endpoints, int n_sweeps = 5) {
  Cable result = cable;
  const int n = static_cast<int>(result.size());
  std::vector<bool> fixed(static_cast<std::size_t>(n), false);
  if (fixed_endpoints && n > 0) {
    fixed.front() = true;
    fixed.back() = true;
  }
  for (int sweep = 0; sweep < n_sweeps; ++sweep) {
    for (int pass = 0; pass < 2; ++pass) {
      const int begin = pass == 0 ? 0 : n - 2;
      const int end = pass == 0 ? n - 1 : -1;
      const int step = pass == 0 ? 1 : -1;
      for (int k = begin; k != end; k += step) {
        const Vec2 seg = result[k + 1] - result[k];
        const Vec2 ref = reference[k + 1] - reference[k];
        const Vec2 q = seg - ref;
        const double qn = norm(q);
        if (qn <= delta_l) continue;
        const Vec2 desired = ref + (delta_l / qn) * q;
        const Vec2 err = seg - desired;
        if (fixed[k] && fixed[k + 1]) continue;
        if (fixed[k]) {
          result[k + 1] -= err;
        } else if (fixed[k + 1]) {
          result[k] += err;
        } else {
          result[k] += 0.5 * err;
          result[k + 1] -= 0.5 * err;
        }
      }
    }
  }
  return result;
}

struct BoundaryQuery {
  Vec2 closest;
  double distance = std::numeric_limits<double>::infinity();
  bool inside = false;
};

BoundaryQuery closest_obstacle_boundary_point(const Vec2& point, const ObstacleSet& obstacles) {
  BoundaryQuery best;
  for (const auto& poly : obstacles.polygons) {
    auto [candidate, d] = closest_point_on_polygon(point, poly);
    const bool inside = point_in_polygon(point, poly);
    if (d < best.distance) {
      best = {candidate, d, inside};
    }
  }
  for (const auto& circle : obstacles.circles) {
    const Vec2 delta = point - circle.center;
    const double n = norm(delta);
    const Vec2 unit = n < EPS ? Vec2{1.0, 0.0} : delta / n;
    const Vec2 candidate = circle.center + circle.radius * unit;
    const double d = std::abs(n - circle.radius);
    if (d < best.distance) {
      best = {candidate, d, n <= circle.radius};
    }
  }
  return best;
}

struct DirectionQuery {
  Vec2 direction;
  double distance = std::numeric_limits<double>::infinity();
  bool inside = false;
  Vec2 boundary;
};

DirectionQuery outward_direction_from_obstacle(const Vec2& point, const ObstacleSet& obstacles) {
  if (obstacles.polygons.empty() && obstacles.circles.empty()) {
    return {{1.0, 0.0}, std::numeric_limits<double>::infinity(), false, point};
  }
  const BoundaryQuery q = closest_obstacle_boundary_point(point, obstacles);
  const Vec2 raw = q.inside ? (q.closest - point) : (point - q.closest);
  return {normalized_or(raw), q.distance, q.inside, q.closest};
}

double signed_distance_to_obstacles(const Vec2& point, const ObstacleSet& obstacles) {
  if (obstacles.polygons.empty() && obstacles.circles.empty()) {
    return std::numeric_limits<double>::infinity();
  }
  const BoundaryQuery q = closest_obstacle_boundary_point(point, obstacles);
  return q.inside ? -q.distance : q.distance;
}

double min_configuration_clearance(const Cable& cable, const ObstacleSet& obstacles, int samples_per_segment) {
  double best = std::numeric_limits<double>::infinity();
  for (const auto& sample : segment_sample_points(cable, samples_per_segment)) {
    best = std::min(best, signed_distance_to_obstacles(sample.point, obstacles));
  }
  return best;
}

bool segment_collision_free(const Vec2& start, const Vec2& end, const ObstacleSet& obstacles) {
  for (const auto& poly : obstacles.polygons) {
    if (segment_intersects_polygon(start, end, poly)) return false;
  }
  for (const auto& circle : obstacles.circles) {
    if (segment_intersects_circle(start, end, circle.center, circle.radius)) return false;
  }
  return true;
}

bool cable_collision_free(const Cable& cable, const ObstacleSet& obstacles) {
  for (std::size_t i = 0; i + 1 < cable.size(); ++i) {
    if (!segment_collision_free(cable[i], cable[i + 1], obstacles)) return false;
  }
  return true;
}

bool configuration_admissible(const Cable& cable, const ObstacleSet& obstacles, double ell_min, double ell_max,
                              const Workspace& workspace) {
  for (const auto& p : cable) {
    if (p.x < workspace.xmin - EPS || p.x > workspace.xmax + EPS || p.y < workspace.ymin - EPS ||
        p.y > workspace.ymax + EPS) {
      return false;
    }
  }
  for (double len : segment_lengths(cable)) {
    if (len < ell_min - EPS || len > ell_max + EPS) return false;
  }
  return cable_collision_free(cable, obstacles);
}

Polygon connection_polygon(const Cable& a, const Cable& b) {
  Polygon poly;
  poly.reserve(a.size() + b.size());
  for (const auto& p : a) poly.push_back(p);
  for (auto it = b.rbegin(); it != b.rend(); ++it) poly.push_back(*it);
  return poly;
}

bool polygon_collision_free(const Polygon& poly, const ObstacleSet& obstacles) {
  for (const auto& obstacle : obstacles.polygons) {
    if (polygon_intersects_polygon(poly, obstacle)) return false;
  }
  for (const auto& circle : obstacles.circles) {
    if (polygon_intersects_circle(poly, circle.center, circle.radius)) return false;
  }
  return true;
}

Vec2 cable_axis_direction(const Cable& cable) {
  const Vec2 axis = cable.back() - cable.front();
  const double n = norm(axis);
  if (n < EPS) return {1.0, 0.0};
  return axis / n;
}

double cable_axis_turn_rad(const Cable& a, const Cable& b) {
  return angle_distance(angle_of(cable_axis_direction(a)), angle_of(cable_axis_direction(b)));
}

bool cable_axis_turn_ok(const Cable& a, const Cable& b, double max_turn_rad) {
  const double max_turn = std::clamp(max_turn_rad, 0.0, PI);
  if (max_turn >= PI - EPS) return true;
  return cable_axis_turn_rad(a, b) <= max_turn + EPS;
}

bool path_free(const Cable& a,
               const Cable& b,
               const ObstacleSet& obstacles,
               int transition_samples,
               bool robot_execution_feasibility = false,
               double robot_tcp_clearance = 0.0,
               double robot_obstacle_clearance = -1.0,
               double robot_swept_tcp_clearance = 0.0,
               double max_cable_axis_turn_rad = PI) {
  if (!cable_axis_turn_ok(a, b, max_cable_axis_turn_rad)) return false;
  if (robot_execution_feasibility) {
    const double tcp_clearance = std::max(0.0, robot_tcp_clearance);
    const double obstacle_clearance =
        robot_obstacle_clearance < 0.0 ? tcp_clearance : std::max(0.0, robot_obstacle_clearance);
    if (!segment_clearance_free(a.front(), b.front(), obstacles, obstacle_clearance)) return false;
    if (!segment_clearance_free(a.back(), b.back(), obstacles, obstacle_clearance)) return false;
    const double min_endpoint_separation = 2.0 * tcp_clearance;
    if (min_endpoint_separation > EPS) {
      for (int s = 0; s <= transition_samples + 1; ++s) {
        const double tau = static_cast<double>(s) / static_cast<double>(transition_samples + 1);
        const Vec2 p0 = (1.0 - tau) * a.front() + tau * b.front();
        const Vec2 p1 = (1.0 - tau) * a.back() + tau * b.back();
        if (distance(p0, p1) < min_endpoint_separation - EPS) return false;
      }
    }
    const double swept_min_separation = 2.0 * std::max(0.0, robot_swept_tcp_clearance);
    if (swept_min_separation > EPS &&
        segment_to_segment_distance(a.front(), b.front(), a.back(), b.back()) < swept_min_separation - EPS) {
      return false;
    }
  }
  if (!polygon_collision_free(connection_polygon(a, b), obstacles)) return false;
  for (std::size_t k = 0; k + 1 < a.size(); ++k) {
    Polygon quad{a[k], a[k + 1], b[k + 1], b[k]};
    if (!polygon_collision_free(quad, obstacles)) return false;
  }
  for (int s = 1; s <= transition_samples; ++s) {
    const double tau = static_cast<double>(s) / static_cast<double>(transition_samples + 1);
    Cable interp(a.size());
    for (std::size_t i = 0; i < a.size(); ++i) interp[i] = (1.0 - tau) * a[i] + tau * b[i];
    if (!cable_collision_free(interp, obstacles)) return false;
  }
  return true;
}

bool path_sequence_free(const std::vector<Cable>& path,
                        const ObstacleSet& obstacles,
                        int transition_samples,
                        bool robot_execution_feasibility = false,
                        double robot_tcp_clearance = 0.0,
                        double robot_obstacle_clearance = -1.0,
                        double robot_swept_tcp_clearance = 0.0,
                        double max_cable_axis_turn_rad = PI) {
  if (path.size() <= 1) return false;
  for (std::size_t k = 0; k + 1 < path.size(); ++k) {
    if (!path_free(path[k], path[k + 1], obstacles, transition_samples, robot_execution_feasibility,
                   robot_tcp_clearance, robot_obstacle_clearance, robot_swept_tcp_clearance,
                   max_cable_axis_turn_rad)) {
      return false;
    }
  }
  return true;
}

bool endpoint_correspondence_overlaps(const Cable& a, const Cable& b) {
  if (a.size() < 2 || b.size() < 2) return false;
  return segments_intersect(a.front(), b.front(), a.back(), b.back());
}

double endpoint_motion_cost(const Cable& a, const Cable& b) {
  return distance(a.front(), b.front()) + distance(a.back(), b.back());
}

constexpr Vec2 ROBOT_LEFT_BASE_2D{-0.25, 0.0};
constexpr Vec2 ROBOT_RIGHT_BASE_2D{1.25, 0.0};
constexpr double ROBOT_BASE_ENDPOINT_REORDER_MARGIN = 0.02;
constexpr double ROBOT_BASE_TCP_PATH_CLEARANCE = 0.055;
constexpr double ROBOT_BASE_ARM_CLEARANCE = 0.070;
constexpr double ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT = 0.0;
constexpr int ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT = 8;
constexpr double ROBOT_TCP_VECTOR_SECTOR_HALF_WIDTH = PI / 2.0;
constexpr double ROBOT_TCP_VECTOR_SECTOR_MAX_ROTATION = 10.0 * PI / 180.0;

bool endpoint0_is_left_robot(const Cable& cable) {
  const double natural = distance(cable.front(), ROBOT_LEFT_BASE_2D) + distance(cable.back(), ROBOT_RIGHT_BASE_2D);
  const double swapped = distance(cable.back(), ROBOT_LEFT_BASE_2D) + distance(cable.front(), ROBOT_RIGHT_BASE_2D);
  return natural <= swapped;
}

std::pair<Vec2, Vec2> robot_assigned_endpoints(const Cable& cable, bool endpoint0_left) {
  if (endpoint0_left) return {cable.front(), cable.back()};
  return {cable.back(), cable.front()};
}

double robot_base_endpoint_assignment_score(const Cable& previous, const Cable& candidate, bool endpoint0_left) {
  const auto [left_prev, right_prev] = robot_assigned_endpoints(previous, endpoint0_left);
  const auto [left_next, right_next] = robot_assigned_endpoints(candidate, endpoint0_left);

  const double travel = distance(left_prev, left_next) + distance(right_prev, right_next);
  const double reach = distance(left_next, ROBOT_LEFT_BASE_2D) + distance(right_next, ROBOT_RIGHT_BASE_2D);
  const double tcp_path_clearance = segment_to_segment_distance(left_prev, left_next, right_prev, right_next);
  const double arm_clearance =
      segment_to_segment_distance(ROBOT_LEFT_BASE_2D, left_next, ROBOT_RIGHT_BASE_2D, right_next);

  const Vec2 base_axis = normalized_or(ROBOT_RIGHT_BASE_2D - ROBOT_LEFT_BASE_2D, {1.0, 0.0});
  const Vec2 prev_axis = normalized_or(right_prev - left_prev, base_axis);
  const Vec2 next_axis = normalized_or(right_next - left_next, prev_axis);
  const Vec2 left_reach_axis = normalized_or(left_next - ROBOT_LEFT_BASE_2D, base_axis);
  const Vec2 right_reach_axis = normalized_or(right_next - ROBOT_RIGHT_BASE_2D, {-base_axis.x, -base_axis.y});

  const double path_cross_penalty = segments_intersect(left_prev, left_next, right_prev, right_next) ? 1.0 : 0.0;
  const double arm_cross_penalty =
      segments_intersect(ROBOT_LEFT_BASE_2D, left_next, ROBOT_RIGHT_BASE_2D, right_next) ? 1.0 : 0.0;
  const double tcp_clearance_penalty = std::max(0.0, ROBOT_BASE_TCP_PATH_CLEARANCE - tcp_path_clearance);
  const double arm_clearance_penalty = std::max(0.0, ROBOT_BASE_ARM_CLEARANCE - arm_clearance);
  const double direction_flip_penalty = std::max(0.0, 1.0 - dot(prev_axis, next_axis));
  const double base_direction_penalty = std::max(0.0, -dot(next_axis, base_axis));
  const Vec2 negative_base_axis{-base_axis.x, -base_axis.y};
  const double reach_direction_penalty =
      std::max(0.0, -dot(left_reach_axis, base_axis)) + std::max(0.0, -dot(right_reach_axis, negative_base_axis));

  return travel + 0.08 * reach + 1.20 * path_cross_penalty + 0.90 * arm_cross_penalty +
         8.0 * tcp_clearance_penalty + 6.0 * arm_clearance_penalty + 0.20 * direction_flip_penalty +
         0.25 * base_direction_penalty + 0.15 * reach_direction_penalty;
}

bool robot_base_reversal_is_preferred(const Cable& previous,
                                      const Cable& candidate,
                                      bool endpoint0_left,
                                      double margin = ROBOT_BASE_ENDPOINT_REORDER_MARGIN) {
  Cable reversed = candidate;
  std::reverse(reversed.begin(), reversed.end());
  const double ordered_score = robot_base_endpoint_assignment_score(previous, candidate, endpoint0_left);
  const double reversed_score = robot_base_endpoint_assignment_score(previous, reversed, endpoint0_left);
  return reversed_score + std::max(0.0, margin) < ordered_score;
}

double robot_base_best_endpoint_assignment_score(const Cable& previous, const Cable& candidate) {
  return std::min(robot_base_endpoint_assignment_score(previous, candidate, true),
                  robot_base_endpoint_assignment_score(previous, candidate, false));
}

bool robot_base_reversal_is_preferred_any_assignment(
    const Cable& previous,
    const Cable& candidate,
    double margin = ROBOT_BASE_ENDPOINT_REORDER_MARGIN) {
  Cable reversed = candidate;
  std::reverse(reversed.begin(), reversed.end());
  const double ordered_score = robot_base_best_endpoint_assignment_score(previous, candidate);
  const double reversed_score = robot_base_best_endpoint_assignment_score(previous, reversed);
  return reversed_score + std::max(0.0, margin) < ordered_score;
}

double robot_base_vector_frontness(double reach_angle) {
  const double front_distance = angle_distance(reach_angle, PI / 2.0);
  return std::clamp((PI / 2.0 - front_distance) / (PI / 2.0), 0.0, 1.0);
}

double robot_base_vector_sector_center(double reach_angle, bool is_right_robot) {
  constexpr double max_sector_rotation = 25.0 * PI / 180.0;
  const double rotation = max_sector_rotation * robot_base_vector_frontness(reach_angle);
  // When the right robot reaches nearly straight forward, rotate its preferred
  // endpoint-vector sector clockwise.  The left robot uses the mirrored rule.
  return wrap_angle(reach_angle + (is_right_robot ? -rotation : rotation));
}

double robot_base_vector_sector_excess(double tcp_direction, double reach_angle, bool is_right_robot) {
  const double sector_center = robot_base_vector_sector_center(reach_angle, is_right_robot);
  return std::max(0.0, angle_distance(tcp_direction, sector_center) - PI / 2.0);
}

double robot_base_vector_sector_score(const Cable& previous,
                                      const Cable& candidate,
                                      bool endpoint0_left) {
  const auto [left_prev, right_prev] = robot_assigned_endpoints(previous, endpoint0_left);
  const auto [left_next, right_next] = robot_assigned_endpoints(candidate, endpoint0_left);
  const double previous_tcp_angle = angle_of(right_prev - left_prev);
  const double next_tcp_angle = angle_of(right_next - left_next);
  const double tcp_turn = angle_distance(next_tcp_angle, previous_tcp_angle);

  const double left_reach_angle = angle_of(left_next - ROBOT_LEFT_BASE_2D);
  const double right_reach_angle = angle_of(right_next - ROBOT_RIGHT_BASE_2D);
  const double left_sector_excess = robot_base_vector_sector_excess(next_tcp_angle, left_reach_angle, false);
  const double right_sector_excess =
      robot_base_vector_sector_excess(wrap_angle(next_tcp_angle + PI), right_reach_angle, true);

  // TCP-vector continuity is the main rule, but the allowed vector sector is
  // rotated by each robot's base-to-TCP reach direction.  This catches cases
  // where the cable endpoint order would force a robot to approach from the
  // wrong side even though the geometric cable profile is valid.
  return tcp_turn + left_sector_excess + right_sector_excess;
}

bool robot_base_vector_reversal_is_preferred_any_assignment(const Cable& previous,
                                                           const Cable& candidate,
                                                           double max_turn_rad,
                                                           double margin_rad) {
  Cable reversed = candidate;
  std::reverse(reversed.begin(), reversed.end());
  const double max_turn = std::max(0.0, max_turn_rad);
  const double margin = std::max(0.0, margin_rad);
  for (bool endpoint0_left : {true, false}) {
    const double ordered_score = robot_base_vector_sector_score(previous, candidate, endpoint0_left);
    const double reversed_score = robot_base_vector_sector_score(previous, reversed, endpoint0_left);
    const bool ordered_ok = ordered_score <= max_turn + EPS;
    const bool reversed_ok = reversed_score <= max_turn + EPS;
    if ((!ordered_ok && reversed_ok) || (reversed_score + margin < ordered_score)) {
      return true;
    }
  }
  return false;
}

double robot_tcp_vector_sector_center(const Vec2& left_tcp, const Vec2& right_tcp) {
  const double left_reach_angle = angle_of(left_tcp - ROBOT_LEFT_BASE_2D);
  const double right_reach_angle = angle_of(right_tcp - ROBOT_RIGHT_BASE_2D);
  const double left_rotation = ROBOT_TCP_VECTOR_SECTOR_MAX_ROTATION * robot_base_vector_frontness(left_reach_angle);
  const double right_rotation = ROBOT_TCP_VECTOR_SECTOR_MAX_ROTATION * robot_base_vector_frontness(right_reach_angle);
  // The primary vector is right TCP -> left TCP, whose nominal feasible sector
  // is the left-facing half-plane centered at 180 deg.  Base-to-TCP reach
  // rotates this sector slightly: right-front reach rotates clockwise, while
  // left-front reach applies the mirrored counter-clockwise adjustment.
  return wrap_angle(PI - right_rotation + left_rotation);
}

double robot_tcp_vector_sector_endpoint_score(const Vec2& left_tcp, const Vec2& right_tcp) {
  const double tcp_direction_right_to_left = angle_of(left_tcp - right_tcp);
  const double sector_center = robot_tcp_vector_sector_center(left_tcp, right_tcp);
  return std::max(0.0, angle_distance(tcp_direction_right_to_left, sector_center) -
                           ROBOT_TCP_VECTOR_SECTOR_HALF_WIDTH);
}

double robot_tcp_vector_sector_score(const Cable& candidate, bool endpoint0_left) {
  const auto [left_tcp, right_tcp] = robot_assigned_endpoints(candidate, endpoint0_left);
  return robot_tcp_vector_sector_endpoint_score(left_tcp, right_tcp);
}

bool robot_tcp_vector_sector_config_ok(const Cable& candidate, bool endpoint0_left, double margin_rad) {
  return robot_tcp_vector_sector_score(candidate, endpoint0_left) <= std::max(0.0, margin_rad) + EPS;
}

bool robot_tcp_vector_sector_transition_ok(const Cable& previous,
                                           const Cable& candidate,
                                           bool endpoint0_left,
                                           double margin_rad,
                                           int transition_samples) {
  const auto [left_prev, right_prev] = robot_assigned_endpoints(previous, endpoint0_left);
  const auto [left_next, right_next] = robot_assigned_endpoints(candidate, endpoint0_left);
  const int samples = std::max(1, transition_samples);
  const double margin = std::max(0.0, margin_rad);
  for (int s = 0; s <= samples; ++s) {
    const double tau = static_cast<double>(s) / static_cast<double>(samples);
    const Vec2 left_tcp = (1.0 - tau) * left_prev + tau * left_next;
    const Vec2 right_tcp = (1.0 - tau) * right_prev + tau * right_next;
    if (robot_tcp_vector_sector_endpoint_score(left_tcp, right_tcp) > margin + EPS) return false;
  }
  return true;
}

bool robot_tcp_vector_sector_path_sequence_ok(const std::vector<Cable>& path,
                                              bool endpoint0_left,
                                              double margin_rad,
                                              int transition_samples) {
  if (path.size() < 2) return true;
  if (!robot_tcp_vector_sector_config_ok(path.front(), endpoint0_left, margin_rad)) return false;
  for (std::size_t k = 0; k + 1 < path.size(); ++k) {
    if (!robot_tcp_vector_sector_transition_ok(path[k], path[k + 1], endpoint0_left, margin_rad,
                                               transition_samples)) {
      return false;
    }
  }
  return true;
}

double robot_tcp_vector_sector_path_score(const std::vector<Cable>& path,
                                          std::size_t start_index,
                                          std::size_t end_index,
                                          bool endpoint0_left) {
  double score = 0.0;
  for (std::size_t k = start_index + 1; k <= end_index; ++k) {
    score += robot_tcp_vector_sector_score(path[k], endpoint0_left);
  }
  return score;
}

bool robot_tcp_vector_sector_reversal_is_preferred(const Cable& previous,
                                                   const Cable& candidate,
                                                   double margin_rad,
                                                   bool endpoint0_left) {
  (void)previous;
  Cable reversed = candidate;
  std::reverse(reversed.begin(), reversed.end());
  const double margin = std::max(0.0, margin_rad);
  const double ordered_score = robot_tcp_vector_sector_score(candidate, endpoint0_left);
  const double reversed_score = robot_tcp_vector_sector_score(reversed, endpoint0_left);
  const bool ordered_ok = ordered_score <= margin + EPS;
  const bool reversed_ok = reversed_score <= margin + EPS;
  return (!ordered_ok && reversed_ok) || (reversed_score + margin < ordered_score);
}

double robot_base_vector_path_score(const std::vector<Cable>& path,
                                    std::size_t start_index,
                                    std::size_t end_index,
                                    bool endpoint0_left) {
  double score = 0.0;
  for (std::size_t k = start_index; k < end_index; ++k) {
    score += robot_base_vector_sector_score(path[k], path[k + 1], endpoint0_left);
  }
  return score;
}

double robot_base_path_assignment_score(const std::vector<Cable>& path,
                                        std::size_t start_index,
                                        std::size_t end_index,
                                        bool endpoint0_left) {
  double score = 0.0;
  for (std::size_t k = start_index; k < end_index; ++k) {
    score += robot_base_endpoint_assignment_score(path[k], path[k + 1], endpoint0_left);
  }
  return score;
}

double robot_base_transition_min_arm_clearance(const Cable& previous,
                                               const Cable& candidate,
                                               bool endpoint0_left,
                                               int transition_samples = ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT) {
  const auto [left_prev, right_prev] = robot_assigned_endpoints(previous, endpoint0_left);
  const auto [left_next, right_next] = robot_assigned_endpoints(candidate, endpoint0_left);
  double min_clearance = std::numeric_limits<double>::infinity();
  const int samples = std::max(1, transition_samples);
  for (int s = 0; s <= samples; ++s) {
    const double tau = static_cast<double>(s) / static_cast<double>(samples);
    const Vec2 left_tcp = (1.0 - tau) * left_prev + tau * left_next;
    const Vec2 right_tcp = (1.0 - tau) * right_prev + tau * right_next;
    min_clearance = std::min(
        min_clearance,
        segment_to_segment_distance(ROBOT_LEFT_BASE_2D, left_tcp, ROBOT_RIGHT_BASE_2D, right_tcp));
  }
  return min_clearance;
}

bool robot_base_transition_clearance_ok(const Cable& previous,
                                        const Cable& candidate,
                                        bool endpoint0_left,
                                        double arm_clearance = ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT,
                                        int transition_samples = ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT) {
  return robot_base_transition_min_arm_clearance(previous, candidate, endpoint0_left, transition_samples) >=
         std::max(0.0, arm_clearance) - EPS;
}

bool robot_base_transition_clearance_ok_either_assignment(
    const Cable& previous,
    const Cable& candidate,
    double arm_clearance = ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT,
    int transition_samples = ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT) {
  return robot_base_transition_clearance_ok(previous, candidate, true, arm_clearance, transition_samples) ||
         robot_base_transition_clearance_ok(previous, candidate, false, arm_clearance, transition_samples);
}

bool robot_base_path_sequence_clearance_ok(const std::vector<Cable>& path,
                                           bool endpoint0_left,
                                           double arm_clearance = ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT,
                                           int transition_samples = ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT) {
  if (path.size() < 2) return true;
  for (std::size_t k = 0; k + 1 < path.size(); ++k) {
    if (!robot_base_transition_clearance_ok(path[k], path[k + 1], endpoint0_left, arm_clearance,
                                            transition_samples)) {
      return false;
    }
  }
  return true;
}

bool robot_base_path_sequence_clearance_ok_either_assignment(
    const std::vector<Cable>& path,
    double arm_clearance = ROBOT_BASE_TRANSITION_ARM_CLEARANCE_DEFAULT,
    int transition_samples = ROBOT_BASE_TRANSITION_SAMPLES_DEFAULT) {
  return robot_base_path_sequence_clearance_ok(path, true, arm_clearance, transition_samples) ||
         robot_base_path_sequence_clearance_ok(path, false, arm_clearance, transition_samples);
}

Vec2 cable_axis(const Cable& cable) {
  const Vec2 axis = cable.back() - cable.front();
  const double n = norm(axis);
  if (n < EPS) return {1.0, 0.0};
  return axis / n;
}

bool endpoint_tangent_reversal_is_preferred(const Cable& previous,
                                            const Cable& candidate,
                                            double tangent_margin) {
  if (tangent_margin < 0.0) return false;
  const double ordered_dot = dot(cable_axis(previous), cable_axis(candidate));
  const double reversed_dot = -ordered_dot;
  return reversed_dot > ordered_dot + tangent_margin;
}

bool endpoint_reversal_is_cheaply_preferred(const Cable& previous,
                                            const Cable& candidate,
                                            double endpoint_reorder_cost_ratio,
                                            double endpoint_reorder_tangent_margin = 0.05,
                                            bool robot_base_aware_endpoint_reorder = false,
                                            bool robot_base_vector_endpoint_reorder = false,
                                            bool robot_tcp_vector_sector_reorder = false,
                                            double robot_base_vector_max_turn_rad = 100.0 * PI / 180.0,
                                            double robot_base_vector_margin_rad = 5.0 * PI / 180.0,
                                            double robot_tcp_vector_sector_margin_rad = 5.0 * PI / 180.0,
                                            bool endpoint0_left = true,
                                            double max_cable_axis_turn_rad = PI) {
  if (previous.size() < 2 || candidate.size() < 2) return false;
  Cable reversed = candidate;
  std::reverse(reversed.begin(), reversed.end());

  const bool ordered_axis_ok = cable_axis_turn_ok(previous, candidate, max_cable_axis_turn_rad);
  const bool reversed_axis_ok = cable_axis_turn_ok(previous, reversed, max_cable_axis_turn_rad);
  if (!ordered_axis_ok && reversed_axis_ok) return true;
  if (ordered_axis_ok && !reversed_axis_ok) return false;

  const bool ordered_overlaps = endpoint_correspondence_overlaps(previous, candidate);
  const bool reversed_overlaps = endpoint_correspondence_overlaps(previous, reversed);
  if (ordered_overlaps && !reversed_overlaps) return true;
  if (!ordered_overlaps && reversed_overlaps) return false;

  if (endpoint_tangent_reversal_is_preferred(previous, candidate, endpoint_reorder_tangent_margin)) return true;
  if (robot_base_aware_endpoint_reorder &&
      (robot_base_reversal_is_preferred(previous, candidate, endpoint0_left) ||
       robot_base_reversal_is_preferred_any_assignment(previous, candidate))) {
    return true;
  }
  if (robot_base_vector_endpoint_reorder &&
      robot_base_vector_reversal_is_preferred_any_assignment(previous, candidate, robot_base_vector_max_turn_rad,
                                                            robot_base_vector_margin_rad)) {
    return true;
  }
  if (robot_tcp_vector_sector_reorder &&
      robot_tcp_vector_sector_reversal_is_preferred(previous, candidate, robot_tcp_vector_sector_margin_rad,
                                                    endpoint0_left)) {
    return true;
  }
  const double old_cost = endpoint_motion_cost(previous, candidate);
  const double reversed_cost = endpoint_motion_cost(previous, reversed);
  return reversed_cost + EPS < endpoint_reorder_cost_ratio * old_cost;
}

bool endpoint_reversal_is_preferred(const Cable& previous,
                                    const Cable& candidate,
                                    bool candidate_valid,
                                    double endpoint_reorder_cost_ratio,
                                    double endpoint_reorder_tangent_margin = 0.05,
                                    bool robot_base_aware_endpoint_reorder = false,
                                    bool robot_base_vector_endpoint_reorder = false,
                                    bool robot_tcp_vector_sector_reorder = false,
                                    double robot_base_vector_max_turn_rad = 100.0 * PI / 180.0,
                                    double robot_base_vector_margin_rad = 5.0 * PI / 180.0,
                                    double robot_tcp_vector_sector_margin_rad = 5.0 * PI / 180.0,
                                    bool endpoint0_left = true,
                                    double max_cable_axis_turn_rad = PI) {
  if (!candidate_valid) return true;
  return endpoint_reversal_is_cheaply_preferred(
      previous, candidate, endpoint_reorder_cost_ratio, endpoint_reorder_tangent_margin,
      robot_base_aware_endpoint_reorder, robot_base_vector_endpoint_reorder, robot_tcp_vector_sector_reorder,
      robot_base_vector_max_turn_rad, robot_base_vector_margin_rad, robot_tcp_vector_sector_margin_rad,
      endpoint0_left, max_cable_axis_turn_rad);
}

std::pair<std::vector<Cable>, int> reorder_endpoint_correspondence(
    const std::vector<Cable>& path,
    const ObstacleSet& obstacles,
    int transition_samples,
    bool robot_execution_feasibility = false,
    double robot_tcp_clearance = 0.0,
    double robot_obstacle_clearance = -1.0,
    double robot_swept_tcp_clearance = 0.0,
    bool preserve_terminal_order = true,
    double endpoint_reorder_cost_ratio = 0.75,
    double endpoint_reorder_tangent_margin = 0.05,
    bool robot_base_aware_endpoint_reorder = false,
    bool robot_base_vector_endpoint_reorder = false,
    bool robot_tcp_vector_sector_reorder = false,
    bool robot_tcp_vector_sector_feasibility = false,
    double robot_base_vector_max_turn_rad = 100.0 * PI / 180.0,
    double robot_base_vector_margin_rad = 5.0 * PI / 180.0,
    double robot_tcp_vector_sector_margin_rad = 5.0 * PI / 180.0,
    double max_cable_axis_turn_rad = PI) {
  if (path.empty()) return {{}, 0};
  std::vector<Cable> ordered;
  ordered.reserve(path.size());
  ordered.push_back(path.front());
  int reversals = 0;
  for (std::size_t i = 1; i < path.size(); ++i) {
    Cable candidate = path[i];
    const bool is_terminal = i + 1 == path.size();
    const bool can_reverse = !(preserve_terminal_order && is_terminal);
    auto transition_valid = [&](const Cable& target) {
      return path_free(ordered.back(), target, obstacles, transition_samples, robot_execution_feasibility,
                       robot_tcp_clearance, robot_obstacle_clearance, robot_swept_tcp_clearance,
                       max_cable_axis_turn_rad) &&
             (!robot_base_aware_endpoint_reorder ||
              robot_base_transition_clearance_ok_either_assignment(ordered.back(), target)) &&
             (!robot_tcp_vector_sector_feasibility ||
              robot_tcp_vector_sector_transition_ok(ordered.back(), target, endpoint0_is_left_robot(ordered.front()),
                                                    robot_tcp_vector_sector_margin_rad, transition_samples));
    };
    if (can_reverse) {
      Cable reversed = candidate;
      std::reverse(reversed.begin(), reversed.end());
      const bool prefer_reversed = endpoint_reversal_is_cheaply_preferred(
          ordered.back(), candidate, endpoint_reorder_cost_ratio, endpoint_reorder_tangent_margin,
          robot_base_aware_endpoint_reorder, robot_base_vector_endpoint_reorder, robot_tcp_vector_sector_reorder,
          robot_base_vector_max_turn_rad, robot_base_vector_margin_rad, robot_tcp_vector_sector_margin_rad,
          endpoint0_is_left_robot(ordered.front()), max_cable_axis_turn_rad);

      if (prefer_reversed) {
        if (transition_valid(reversed)) {
          candidate = std::move(reversed);
          ++reversals;
        }
      } else if (!transition_valid(candidate) && transition_valid(reversed)) {
        candidate = std::move(reversed);
        ++reversals;
      }
    }
    ordered.push_back(std::move(candidate));
  }
  return {ordered, reversals};
}

std::vector<Vec2> cable_normals(const Cable& cable) {
  std::vector<Vec2> normals(cable.size(), {0.0, 1.0});
  if (cable.size() < 2) return normals;
  for (std::size_t i = 0; i < cable.size(); ++i) {
    Vec2 tangent;
    if (i == 0) {
      tangent = cable[1] - cable[0];
    } else if (i + 1 == cable.size()) {
      tangent = cable[i] - cable[i - 1];
    } else {
      tangent = cable[i + 1] - cable[i - 1];
    }
    normals[i] = normalized_or({-tangent.y, tangent.x}, {0.0, 1.0});
  }
  return normals;
}

std::vector<Vec2> configuration_forces(const Cable& cable, const Cable& target, const ObstacleSet& obstacles,
                                       const PlannerParams& params) {
  std::vector<Vec2> total(cable.size(), {0.0, 0.0});
  for (std::size_t i = 0; i < cable.size(); ++i) {
    total[i] += params.k_att * (target[i] - cable[i]);
  }
  if (obstacles.polygons.empty() && obstacles.circles.empty()) return total;

  const double sigma_sq = std::max(params.sigma * params.sigma, EPS);
  const double rep_inside_strength = params.beta / std::max(params.sigma, EPS);
  for (const auto& sample : segment_sample_points(cable, params.segment_samples)) {
    const auto q = outward_direction_from_obstacle(sample.point, obstacles);
    double strength = 0.0;
    if (q.inside) {
      strength = rep_inside_strength;
    } else {
      const double exponent = std::exp(-(q.distance * q.distance) / (2.0 * params.sigma * params.sigma));
      strength = params.beta * q.distance / sigma_sq * exponent;
    }
    const Vec2 f = strength * q.direction;
    total[static_cast<std::size_t>(sample.k)] += (1.0 - sample.rho) * f;
    total[static_cast<std::size_t>(sample.k + 1)] += sample.rho * f;
  }
  return total;
}

Cable apf_descent(const Cable& raw, const Cable& target, const ObstacleSet& obstacles, const Workspace& workspace,
                  double ell0, const PlannerParams& params) {
  Cable current = raw;
  for (int step = 0; step < params.apf_steps; ++step) {
    std::vector<Vec2> forces = configuration_forces(current, target, obstacles, params);
    std::vector<Vec2> displacement(current.size());
    double max_step = 0.0;
    for (std::size_t i = 0; i < current.size(); ++i) {
      displacement[i] = params.eta * forces[i];
      max_step = std::max(max_step, norm(displacement[i]));
    }
    if (max_step > params.max_apf_node_step) {
      const double scale = params.max_apf_node_step / max_step;
      for (auto& d : displacement) d = d * scale;
    }
    for (std::size_t i = 0; i < current.size(); ++i) current[i] += displacement[i];
    current = clip_to_workspace(current, workspace);
    current = project_uniform_segment_lengths(current, ell0, 2);
    current = clip_to_workspace(current, workspace);
  }
  return current;
}

struct RelaxationPlane {
  int k = 0;
  double rho = 0.0;
  Vec2 normal;
  Vec2 boundary_point;
};

std::vector<RelaxationPlane> make_relaxation_planes(const Cable& biased, const ObstacleSet& obstacles,
                                                    const PlannerParams& params) {
  std::vector<RelaxationPlane> planes;
  for (const auto& sample : segment_sample_points(biased, params.segment_samples)) {
    const auto q = outward_direction_from_obstacle(sample.point, obstacles);
    planes.push_back({sample.k, sample.rho, q.direction, q.boundary});
  }
  return planes;
}

std::vector<Vec2> relaxation_gradient(const Cable& cable, const Cable& biased, const Cable& reference,
                                      const PlannerParams& params) {
  std::vector<Vec2> grad(cable.size());
  for (std::size_t i = 0; i < cable.size(); ++i) grad[i] = params.tracking_weight * (cable[i] - biased[i]);
  for (std::size_t k = 0; k + 1 < cable.size(); ++k) {
    const Vec2 seg_q = (cable[k + 1] - cable[k]) - (reference[k + 1] - reference[k]);
    grad[k] -= 2.0 * params.lambda_s * seg_q;
    grad[k + 1] += 2.0 * params.lambda_s * seg_q;
  }
  for (std::size_t k = 0; k + 2 < cable.size(); ++k) {
    const Vec2 bend_q =
        (cable[k + 2] - 2.0 * cable[k + 1] + cable[k]) -
        (reference[k + 2] - 2.0 * reference[k + 1] + reference[k]);
    grad[k] += 2.0 * params.lambda_b * bend_q;
    grad[k + 1] -= 4.0 * params.lambda_b * bend_q;
    grad[k + 2] += 2.0 * params.lambda_b * bend_q;
  }
  return grad;
}

Cable project_linearized_obstacle_constraints(const Cable& cable, const std::vector<RelaxationPlane>& planes,
                                              double dsafe, bool fixed_endpoints) {
  Cable result = cable;
  std::vector<bool> fixed(result.size(), false);
  if (fixed_endpoints && !fixed.empty()) {
    fixed.front() = true;
    fixed.back() = true;
  }
  for (const auto& plane : planes) {
    const int k = plane.k;
    const double w0 = 1.0 - plane.rho;
    const double w1 = plane.rho;
    const Vec2 sample = w0 * result[static_cast<std::size_t>(k)] + w1 * result[static_cast<std::size_t>(k + 1)];
    const double gap = dot(plane.normal, sample - plane.boundary_point);
    const double violation = dsafe - gap;
    if (violation <= 0.0) continue;
    const double fw0 = fixed[static_cast<std::size_t>(k)] ? 0.0 : w0;
    const double fw1 = fixed[static_cast<std::size_t>(k + 1)] ? 0.0 : w1;
    const double denom = fw0 * fw0 + fw1 * fw1;
    if (denom < EPS) continue;
    if (fw0 > 0.0) result[static_cast<std::size_t>(k)] += (fw0 / denom) * violation * plane.normal;
    if (fw1 > 0.0) result[static_cast<std::size_t>(k + 1)] += (fw1 / denom) * violation * plane.normal;
  }
  return result;
}

Cable push_sample_clearance(const Cable& cable, const ObstacleSet& obstacles, const PlannerParams& params,
                            bool fixed_endpoints) {
  Cable result = cable;
  if (obstacles.polygons.empty() && obstacles.circles.empty()) return result;
  std::vector<bool> fixed(result.size(), false);
  if (fixed_endpoints && !fixed.empty()) {
    fixed.front() = true;
    fixed.back() = true;
  }
  for (const auto& sample : segment_sample_points(result, params.segment_samples)) {
    const auto q = outward_direction_from_obstacle(sample.point, obstacles);
    const double signed_d = q.inside ? -q.distance : q.distance;
    const double violation = params.dsafe - signed_d;
    if (violation <= 0.0) continue;
    const double w0 = 1.0 - sample.rho;
    const double w1 = sample.rho;
    const double fw0 = fixed[static_cast<std::size_t>(sample.k)] ? 0.0 : w0;
    const double fw1 = fixed[static_cast<std::size_t>(sample.k + 1)] ? 0.0 : w1;
    const double denom = fw0 * fw0 + fw1 * fw1;
    if (denom < EPS) continue;
    if (fw0 > 0.0) result[static_cast<std::size_t>(sample.k)] += (fw0 / denom) * violation * q.direction;
    if (fw1 > 0.0) result[static_cast<std::size_t>(sample.k + 1)] += (fw1 / denom) * violation * q.direction;
  }
  return result;
}

Cable relax_configuration(const Cable& biased, const Cable& reference, const ObstacleSet& obstacles,
                          const Workspace& workspace, const PlannerParams& params) {
  const std::vector<RelaxationPlane> planes = make_relaxation_planes(biased, obstacles, params);
  Cable result = biased;
  for (int iter = 0; iter < params.relax_iterations; ++iter) {
    std::vector<Vec2> grad = relaxation_gradient(result, biased, reference, params);
    if (!grad.empty()) {
      grad.front() = {0.0, 0.0};
      grad.back() = {0.0, 0.0};
    }
    double max_grad = 0.0;
    for (const auto& g : grad) max_grad = std::max(max_grad, norm(g));
    if (max_grad > 2.0) {
      for (auto& g : grad) g = g * (2.0 / max_grad);
    }
    for (std::size_t i = 0; i < result.size(); ++i) result[i] -= params.relax_step * grad[i];
    result.front() = biased.front();
    result.back() = biased.back();
    result = project_linearized_obstacle_constraints(result, planes, params.dsafe, true);
    result.front() = biased.front();
    result.back() = biased.back();
    if (iter % params.exact_clearance_period == 0 || iter == params.relax_iterations - 1) {
      result = push_sample_clearance(result, obstacles, params, true);
      result.front() = biased.front();
      result.back() = biased.back();
    }
    result = project_segment_vector_deviation(result, reference, params.delta_l, true, 3);
    result.front() = biased.front();
    result.back() = biased.back();
    result = clip_to_workspace(result, workspace);
    result.front() = clip_to_workspace(Cable{result.front()}, workspace).front();
    result.back() = clip_to_workspace(Cable{result.back()}, workspace).front();
  }
  return result;
}

class GCRPlanner {
 public:
  GCRPlanner(Scenario scenario, PlannerParams params)
      : scenario_(std::move(scenario)), params_(params), rng_(static_cast<std::uint64_t>(scenario_.seed)) {
    params_.epsilon = scenario_.epsilon;
    ell0_ = nominal_segment_length(scenario_.start);
    ell_min_ = params_.ell_min_ratio * ell0_;
    ell_max_ = params_.ell_max_ratio * ell0_;
  }

  PlanResult plan() {
    using Clock = std::chrono::steady_clock;
    const auto start_time = Clock::now();
    SearchTree ts{"start", {TreeNode{scenario_.start, -1, 0.0}}};
    SearchTree tg{"goal", {TreeNode{scenario_.goal, -1, 0.0}}};
    bool success = false;
    bool timed_out = false;
    int final_iteration = 0;
    std::vector<Cable> raw_path;
    std::vector<Cable> refined_path;
    endpoint_reversals_ = 0;

    for (int iteration = 1; iteration <= params_.max_iterations; ++iteration) {
      final_iteration = iteration;
      if (params_.has_time_limit) {
        const double elapsed = std::chrono::duration<double>(Clock::now() - start_time).count();
        if (elapsed >= params_.scenario_time_limit_s) {
          timed_out = true;
          final_iteration = iteration - 1;
          break;
        }
      }

      SearchTree* active = nullptr;
      SearchTree* opposite = nullptr;
      const Cable* target = nullptr;
      const Cable* reference = nullptr;
      if (iteration % 2 == 1) {
        active = &ts;
        opposite = &tg;
        target = &scenario_.goal;
        reference = &scenario_.start;
      } else {
        active = &tg;
        opposite = &ts;
        target = &scenario_.start;
        reference = &scenario_.goal;
      }

      auto [expanded, new_index] = expand_tree(*active, *target, *reference);
      if (!expanded || new_index < 0) continue;

      const Cable& new_config = active->nodes[static_cast<std::size_t>(new_index)].config;
      auto [conn_index, conn_distance] = nearest_index(*opposite, new_config);
      if (conn_distance <= params_.epsilon &&
          path_free(new_config, opposite->nodes[static_cast<std::size_t>(conn_index)].config, scenario_.obstacles,
                    params_.transition_samples, params_.robot_execution_feasibility, params_.robot_tcp_clearance,
                    robot_obstacle_clearance_value(), params_.robot_swept_tcp_clearance,
                    params_.max_cable_axis_turn_rad)) {
        int meet_s = -1;
        int meet_g = -1;
        if (active->name == "start") {
          meet_s = new_index;
          meet_g = conn_index;
        } else {
          meet_s = conn_index;
          meet_g = new_index;
        }
        raw_path = trace_root_to(ts, meet_s);
        std::vector<Cable> goal_path = trace_to_root(tg, meet_g);
        raw_path.insert(raw_path.end(), goal_path.begin(), goal_path.end());
        std::vector<Cable> original_raw_path = raw_path;
        if (params_.endpoint_reorder) {
          auto reordered = reorder_endpoint_correspondence(
              raw_path, scenario_.obstacles, params_.transition_samples, params_.robot_execution_feasibility,
              params_.robot_tcp_clearance, robot_obstacle_clearance_value(), params_.robot_swept_tcp_clearance,
              params_.preserve_terminal_order, params_.endpoint_reorder_cost_ratio,
              params_.endpoint_reorder_tangent_margin, params_.robot_base_aware_endpoint_reorder,
              params_.robot_base_vector_endpoint_reorder, params_.robot_tcp_vector_sector_reorder,
              params_.robot_tcp_vector_sector_feasibility, params_.robot_base_vector_max_turn_rad,
              params_.robot_base_vector_margin_rad,
              params_.robot_tcp_vector_sector_margin_rad, params_.max_cable_axis_turn_rad);
          raw_path = std::move(reordered.first);
          endpoint_reversals_ = reordered.second;
          const bool raw_robot_base_ok =
              !params_.robot_base_aware_endpoint_reorder ||
              robot_base_path_sequence_clearance_ok_either_assignment(raw_path, params_.robot_base_arm_clearance,
                                                                       params_.robot_base_transition_samples);
          const bool raw_tcp_sector_ok =
              !params_.robot_tcp_vector_sector_feasibility ||
              robot_tcp_vector_sector_path_sequence_ok(raw_path, endpoint0_is_left_robot(raw_path.front()),
                                                       params_.robot_tcp_vector_sector_margin_rad,
                                                       params_.transition_samples);
          if (!raw_robot_base_ok || !raw_tcp_sector_ok ||
              !path_sequence_free(raw_path, scenario_.obstacles, params_.transition_samples,
                                  params_.robot_execution_feasibility, params_.robot_tcp_clearance,
                                  robot_obstacle_clearance_value(), params_.robot_swept_tcp_clearance,
                                  params_.max_cable_axis_turn_rad)) {
            raw_path = std::move(original_raw_path);
            endpoint_reversals_ = 0;
          }
        }
        if (params_.robot_base_aware_endpoint_reorder &&
            !robot_base_path_sequence_clearance_ok_either_assignment(raw_path, params_.robot_base_arm_clearance,
                                                                     params_.robot_base_transition_samples)) {
          ++reject_no_connect_;
          continue;
        }
        refined_path = refine_path(raw_path);
        const int validation_samples = std::max(params_.transition_samples, params_.shortcut_validation_samples);
        const bool refined_robot_base_ok =
            !params_.robot_base_aware_endpoint_reorder ||
            robot_base_path_sequence_clearance_ok_either_assignment(refined_path, params_.robot_base_arm_clearance,
                                                                    params_.robot_base_transition_samples);
        const bool refined_tcp_sector_ok =
            !params_.robot_tcp_vector_sector_feasibility ||
            robot_tcp_vector_sector_path_sequence_ok(refined_path, endpoint0_is_left_robot(refined_path.front()),
                                                     params_.robot_tcp_vector_sector_margin_rad, validation_samples);
        if (!refined_robot_base_ok || !refined_tcp_sector_ok ||
            !path_sequence_free(refined_path, scenario_.obstacles, validation_samples,
                                params_.robot_execution_feasibility, params_.robot_tcp_clearance,
                                robot_obstacle_clearance_value(), params_.robot_swept_tcp_clearance,
                                params_.max_cable_axis_turn_rad)) {
          refined_path = raw_path;
        }
        const bool final_robot_base_ok =
            !params_.robot_base_aware_endpoint_reorder ||
            robot_base_path_sequence_clearance_ok_either_assignment(refined_path, params_.robot_base_arm_clearance,
                                                                    params_.robot_base_transition_samples);
        const bool final_tcp_sector_ok =
            !params_.robot_tcp_vector_sector_feasibility ||
            robot_tcp_vector_sector_path_sequence_ok(refined_path, endpoint0_is_left_robot(refined_path.front()),
                                                     params_.robot_tcp_vector_sector_margin_rad, validation_samples);
        if (final_robot_base_ok && final_tcp_sector_ok &&
            path_sequence_free(refined_path, scenario_.obstacles, validation_samples,
                               params_.robot_execution_feasibility, params_.robot_tcp_clearance,
                               robot_obstacle_clearance_value(), params_.robot_swept_tcp_clearance,
                               params_.max_cable_axis_turn_rad)) {
          success = true;
          final_iteration = iteration;
          break;
        }
      }
      ++reject_no_connect_;
    }

    const double elapsed = std::chrono::duration<double>(Clock::now() - start_time).count();
    if (!success && !timed_out && params_.has_time_limit && elapsed >= params_.scenario_time_limit_s) timed_out = true;
    Metrics metrics = compute_metrics(success, timed_out, raw_path, refined_path, final_iteration, elapsed, ts, tg);
    return {scenario_, success, timed_out, raw_path, refined_path, final_iteration, elapsed, ts, tg, metrics};
  }

 private:
  Scenario scenario_;
  PlannerParams params_;
  std::mt19937_64 rng_;
  double ell0_ = 0.0;
  double ell_min_ = 0.0;
  double ell_max_ = 0.0;
  int reject_static_ = 0;
  int reject_no_parent_ = 0;
  int reject_no_connect_ = 0;
  int endpoint_reversals_ = 0;

  double robot_obstacle_clearance_value() const {
    return params_.has_robot_obstacle_clearance ? params_.robot_obstacle_clearance : -1.0;
  }

  double uniform(double a = 0.0, double b = 1.0) {
    std::uniform_real_distribution<double> dist(a, b);
    return dist(rng_);
  }

  double normal(double sigma) {
    std::normal_distribution<double> dist(0.0, sigma);
    return dist(rng_);
  }

  int randint(int exclusive_max) {
    std::uniform_int_distribution<int> dist(0, exclusive_max - 1);
    return dist(rng_);
  }

  bool allclose(const Cable& a, const Cable& b, double tol = 1e-8) const {
    if (a.size() != b.size()) return false;
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (std::abs(a[i].x - b[i].x) > tol || std::abs(a[i].y - b[i].y) > tol) return false;
    }
    return true;
  }

  std::pair<bool, int> expand_tree(SearchTree& active, const Cable& target, const Cable& reference) {
    Cable raw = sample_raw_configuration(active, target);
    Cable biased = apf_descent(raw, target, scenario_.obstacles, scenario_.workspace, ell0_, params_);
    Cable relaxed = relax_configuration(biased, reference, scenario_.obstacles, scenario_.workspace, params_);

    if (!configuration_admissible(relaxed, scenario_.obstacles, ell_min_, ell_max_, scenario_.workspace)) {
      ++reject_static_;
      return {false, -1};
    }
    const std::optional<int> parent_index = select_parent(active, relaxed);
    if (!parent_index.has_value()) {
      ++reject_no_parent_;
      return {false, -1};
    }
    const TreeNode& parent = active.nodes[static_cast<std::size_t>(*parent_index)];
    const double d_parent = configuration_distance(parent.config, relaxed);
    const double cost = parent.cost + d_parent + params_.alpha_def * deformation_energy(relaxed, parent.config);
    active.nodes.push_back({relaxed, *parent_index, cost});
    return {true, static_cast<int>(active.nodes.size()) - 1};
  }

  std::optional<int> select_parent(const SearchTree& tree, const Cable& relaxed) const {
    std::vector<std::pair<double, int>> candidates;
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
      const double d = configuration_distance(tree.nodes[i].config, relaxed);
      if (d <= params_.epsilon) candidates.push_back({d, static_cast<int>(i)});
    }
    if (candidates.empty()) return std::nullopt;
    std::sort(candidates.begin(), candidates.end());
    const int n_checks = std::min(params_.max_parent_checks, static_cast<int>(candidates.size()));
    for (int i = 0; i < n_checks; ++i) {
      const int index = candidates[static_cast<std::size_t>(i)].second;
      if (path_free(tree.nodes[static_cast<std::size_t>(index)].config, relaxed, scenario_.obstacles,
                    params_.transition_samples, params_.robot_execution_feasibility, params_.robot_tcp_clearance,
                    robot_obstacle_clearance_value(), params_.robot_swept_tcp_clearance,
                    params_.max_cable_axis_turn_rad)) {
        return index;
      }
    }
    return std::nullopt;
  }

  std::pair<int, double> nearest_index(const SearchTree& tree, const Cable& config) const {
    int best_index = 0;
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < tree.nodes.size(); ++i) {
      const double d = configuration_distance(tree.nodes[i].config, config);
      if (d < best) {
        best = d;
        best_index = static_cast<int>(i);
      }
    }
    return {best_index, best};
  }

  Cable sample_raw_configuration(const SearchTree& active, const Cable& target) {
    const double p = uniform();
    Cable base;
    double noise_scale = 0.0;
    if (p < params_.goal_bias) {
      base = target;
      noise_scale = 0.018;
    } else if (p < params_.goal_bias + params_.tree_bias) {
      std::vector<std::pair<double, int>> target_dist;
      target_dist.reserve(active.nodes.size());
      for (std::size_t i = 0; i < active.nodes.size(); ++i) {
        target_dist.push_back({configuration_distance(active.nodes[i].config, target), static_cast<int>(i)});
      }
      int parent_index = 0;
      if (active.nodes.size() > 4 && uniform() < 0.65) {
        std::sort(target_dist.begin(), target_dist.end());
        const int pool_size = std::max(3, std::min(12, static_cast<int>(target_dist.size())));
        parent_index = target_dist[static_cast<std::size_t>(randint(pool_size))].second;
      } else {
        parent_index = randint(static_cast<int>(active.nodes.size()));
      }
      const Cable& parent = active.nodes[static_cast<std::size_t>(parent_index)].config;
      Cable local_target = local_sampling_target(parent, target);
      const double d = std::max(configuration_distance(parent, local_target), EPS);
      const double alpha = std::min(0.64, std::max(0.16, 0.70 * params_.epsilon / d));
      base.resize(parent.size());
      for (std::size_t i = 0; i < parent.size(); ++i) base[i] = parent[i] + alpha * (local_target[i] - parent[i]);
      noise_scale = 0.012;
    } else {
      return sample_broad_configuration();
    }

    Cable raw = base;
    const std::vector<Vec2> normals = cable_normals(base);
    const double lateral_amp = normal(0.024);
    for (std::size_t i = 0; i < raw.size(); ++i) {
      const double sinusoid = std::sin(PI * static_cast<double>(i) / static_cast<double>(raw.size() - 1));
      raw[i] += Vec2{normal(noise_scale), normal(noise_scale)};
      raw[i] += (lateral_amp * sinusoid) * normals[i];
    }
    raw = clip_to_workspace(raw, scenario_.workspace);
    raw = project_uniform_segment_lengths(raw, ell0_, 5);
    return clip_to_workspace(raw, scenario_.workspace);
  }

  Cable local_sampling_target(const Cable& parent, const Cable& final_target) const {
    const bool forward = allclose(final_target, scenario_.goal);
    if (!scenario_.guide_configs.empty()) {
      const std::vector<Cable>& configs = scenario_.guide_configs;
      std::vector<int> order(configs.size());
      std::iota(order.begin(), order.end(), 0);
      if (!forward) std::reverse(order.begin(), order.end());
      int best_pos = 0;
      double best_d = std::numeric_limits<double>::infinity();
      for (std::size_t pos = 0; pos < order.size(); ++pos) {
        const double d = configuration_distance(parent, configs[static_cast<std::size_t>(order[pos])]);
        if (d < best_d) {
          best_d = d;
          best_pos = static_cast<int>(pos);
        }
      }
      const int next_pos = std::min(best_pos + 1, static_cast<int>(order.size()) - 1);
      return configs[static_cast<std::size_t>(order[static_cast<std::size_t>(next_pos)])];
    }

    if (scenario_.guide_points.size() < 2) return final_target;
    std::vector<Vec2> points = scenario_.guide_points;
    if (!forward) std::reverse(points.begin(), points.end());
    Vec2 center{0.0, 0.0};
    for (const auto& p : parent) center += p;
    center = center / static_cast<double>(parent.size());
    int best = 0;
    double best_d = std::numeric_limits<double>::infinity();
    for (std::size_t i = 0; i < points.size(); ++i) {
      const double d = distance(points[i], center);
      if (d < best_d) {
        best_d = d;
        best = static_cast<int>(i);
      }
    }
    const int next = std::min(best + 1, static_cast<int>(points.size()) - 1);
    const Vec2 shift = points[static_cast<std::size_t>(next)] - center;
    Cable out = parent;
    for (auto& p : out) p += shift;
    return out;
  }

  Cable sample_broad_configuration() {
    const int n = static_cast<int>(scenario_.start.size());
    Cable cable;
    for (int attempt = 0; attempt < 80; ++attempt) {
      cable.clear();
      double angle = uniform(0.0, 2.0 * PI);
      Vec2 p0{uniform(scenario_.workspace.xmin + 0.06, scenario_.workspace.xmax - 0.06),
              uniform(scenario_.workspace.ymin + 0.06, scenario_.workspace.ymax - 0.06)};
      cable.push_back(p0);
      for (int k = 0; k < n - 1; ++k) {
        angle += normal(0.36);
        cable.push_back(cable.back() + ell0_ * Vec2{std::cos(angle), std::sin(angle)});
      }
      Vec2 min_xy = cable.front();
      Vec2 max_xy = cable.front();
      for (const auto& p : cable) {
        min_xy.x = std::min(min_xy.x, p.x);
        min_xy.y = std::min(min_xy.y, p.y);
        max_xy.x = std::max(max_xy.x, p.x);
        max_xy.y = std::max(max_xy.y, p.y);
      }
      Vec2 shift{0.0, 0.0};
      if (min_xy.x < scenario_.workspace.xmin + 0.015) shift.x = scenario_.workspace.xmin + 0.015 - min_xy.x;
      if (max_xy.x > scenario_.workspace.xmax - 0.015) shift.x = scenario_.workspace.xmax - 0.015 - max_xy.x;
      if (min_xy.y < scenario_.workspace.ymin + 0.015) shift.y = scenario_.workspace.ymin + 0.015 - min_xy.y;
      if (max_xy.y > scenario_.workspace.ymax - 0.015) shift.y = scenario_.workspace.ymax - 0.015 - max_xy.y;
      for (auto& p : cable) p += shift;
      cable = clip_to_workspace(cable, scenario_.workspace);
      if (min_configuration_clearance(cable, scenario_.obstacles, params_.segment_samples) > -0.03) return cable;
    }
    return clip_to_workspace(cable, scenario_.workspace);
  }

  std::vector<Cable> refine_path(const std::vector<Cable>& path) {
    if (path.size() <= 2 || !params_.shortcut_refinement) return path;
    std::vector<Cable> refined = path;
    for (int pass = 0; pass < params_.shortcut_passes; ++pass) {
      bool improved = false;
      std::size_t i = 0;
      while (i + 2 < refined.size()) {
        bool accepted = false;
        for (std::size_t j = refined.size() - 1; j > i + 1; --j) {
          double old_cost = 0.0;
          for (std::size_t k = i; k < j; ++k) old_cost += configuration_distance(refined[k], refined[k + 1]);
          if (params_.endpoint_reorder && params_.preserve_terminal_order &&
              endpoint_correspondence_overlaps(refined[i], refined[j])) {
            continue;
          }
          std::vector<std::pair<Cable, bool>> shortcut_candidates;
          const bool can_reverse_shortcut = params_.endpoint_reorder && !params_.preserve_terminal_order;
          const bool prefer_reversed_shortcut =
              can_reverse_shortcut &&
              endpoint_reversal_is_cheaply_preferred(
                  refined[i], refined[j], params_.endpoint_reorder_cost_ratio,
                  params_.endpoint_reorder_tangent_margin, params_.robot_base_aware_endpoint_reorder,
                  params_.robot_base_vector_endpoint_reorder, params_.robot_tcp_vector_sector_reorder,
                  params_.robot_base_vector_max_turn_rad, params_.robot_base_vector_margin_rad,
                  params_.robot_tcp_vector_sector_margin_rad, endpoint0_is_left_robot(refined.front()),
                  params_.max_cable_axis_turn_rad);
          if (prefer_reversed_shortcut) {
            Cable reversed_target = refined[j];
            std::reverse(reversed_target.begin(), reversed_target.end());
            shortcut_candidates.push_back({std::move(reversed_target), true});
            shortcut_candidates.push_back({refined[j], false});
          } else {
            shortcut_candidates.push_back({refined[j], false});
            if (can_reverse_shortcut) {
              Cable reversed_target = refined[j];
              std::reverse(reversed_target.begin(), reversed_target.end());
              shortcut_candidates.push_back({std::move(reversed_target), true});
            }
          }
          for (const auto& candidate_pair : shortcut_candidates) {
            const Cable& shortcut_target = candidate_pair.first;
            const bool reverse_suffix = candidate_pair.second;
            const double new_cost = configuration_distance(refined[i], shortcut_target);
            const double max_shortcut_distance =
                params_.has_shortcut_max_distance_factor ? params_.shortcut_max_distance_factor * params_.epsilon
                                                         : std::numeric_limits<double>::infinity();
            bool robot_base_shortcut_ok = true;
            if (params_.robot_base_aware_endpoint_reorder) {
              const double old_robot_score =
                  std::min(robot_base_path_assignment_score(refined, i, j, true),
                           robot_base_path_assignment_score(refined, i, j, false));
              const double new_robot_score =
                  robot_base_best_endpoint_assignment_score(refined[i], shortcut_target);
              robot_base_shortcut_ok = new_robot_score <= old_robot_score + ROBOT_BASE_ENDPOINT_REORDER_MARGIN &&
                                       robot_base_transition_clearance_ok_either_assignment(
                                           refined[i], shortcut_target, params_.robot_base_arm_clearance,
                                           params_.robot_base_transition_samples);
            }
            if (params_.robot_base_vector_endpoint_reorder) {
              const double old_vector_score =
                  std::min(robot_base_vector_path_score(refined, i, j, true),
                           robot_base_vector_path_score(refined, i, j, false));
              const double new_vector_score =
                  std::min(robot_base_vector_sector_score(refined[i], shortcut_target, true),
                           robot_base_vector_sector_score(refined[i], shortcut_target, false));
              robot_base_shortcut_ok =
                  robot_base_shortcut_ok &&
                  new_vector_score <= old_vector_score + params_.robot_base_vector_margin_rad;
            }
            if (params_.robot_tcp_vector_sector_reorder) {
              const double old_sector_score =
                  std::min(robot_tcp_vector_sector_path_score(refined, i, j, true),
                           robot_tcp_vector_sector_path_score(refined, i, j, false));
              const double new_sector_score =
                  std::min(robot_tcp_vector_sector_score(shortcut_target, true),
                           robot_tcp_vector_sector_score(shortcut_target, false));
              robot_base_shortcut_ok =
                  robot_base_shortcut_ok &&
                  new_sector_score <= old_sector_score + params_.robot_tcp_vector_sector_margin_rad;
            }
            if (params_.robot_tcp_vector_sector_feasibility) {
              robot_base_shortcut_ok =
                  robot_base_shortcut_ok &&
                  robot_tcp_vector_sector_transition_ok(refined[i], shortcut_target,
                                                        endpoint0_is_left_robot(refined.front()),
                                                        params_.robot_tcp_vector_sector_margin_rad,
                                                        std::max(params_.transition_samples,
                                                                 params_.shortcut_validation_samples));
            }
            if (!(robot_base_shortcut_ok && new_cost <= max_shortcut_distance && new_cost < old_cost)) {
              continue;
            }
            std::vector<Cable> next;
            next.insert(next.end(), refined.begin(), refined.begin() + static_cast<std::ptrdiff_t>(i + 1));
            for (auto it = refined.begin() + static_cast<std::ptrdiff_t>(j); it != refined.end(); ++it) {
              Cable config = *it;
              if (reverse_suffix) std::reverse(config.begin(), config.end());
              next.push_back(std::move(config));
            }
            if (params_.robot_base_aware_endpoint_reorder &&
                !robot_base_path_sequence_clearance_ok_either_assignment(next, params_.robot_base_arm_clearance,
                                                                         params_.robot_base_transition_samples)) {
              continue;
            }
            if (params_.robot_tcp_vector_sector_feasibility &&
                !robot_tcp_vector_sector_path_sequence_ok(next, endpoint0_is_left_robot(next.front()),
                                                          params_.robot_tcp_vector_sector_margin_rad,
                                                          std::max(params_.transition_samples,
                                                                   params_.shortcut_validation_samples))) {
              continue;
            }
            if (!path_free(refined[i], shortcut_target, scenario_.obstacles,
                           std::max(params_.transition_samples, params_.shortcut_validation_samples),
                           params_.robot_execution_feasibility, params_.robot_tcp_clearance,
                           robot_obstacle_clearance_value(), params_.robot_swept_tcp_clearance,
                           params_.max_cable_axis_turn_rad)) {
              continue;
            }
            if (reverse_suffix) ++endpoint_reversals_;
            refined.swap(next);
            improved = true;
            accepted = true;
            break;
          }
          if (accepted) break;
        }
        if (!accepted) ++i;
      }
      if (!improved) break;
    }
    return refined;
  }

  std::vector<Cable> trace_root_to(const SearchTree& tree, int index) const {
    std::vector<Cable> path;
    int cur = index;
    while (cur >= 0) {
      const TreeNode& node = tree.nodes[static_cast<std::size_t>(cur)];
      path.push_back(node.config);
      cur = node.parent;
    }
    std::reverse(path.begin(), path.end());
    return path;
  }

  std::vector<Cable> trace_to_root(const SearchTree& tree, int index) const {
    std::vector<Cable> path;
    int cur = index;
    while (cur >= 0) {
      const TreeNode& node = tree.nodes[static_cast<std::size_t>(cur)];
      path.push_back(node.config);
      cur = node.parent;
    }
    return path;
  }

  Metrics compute_metrics(bool success, bool timed_out, const std::vector<Cable>& raw_path,
                          const std::vector<Cable>& path, int iterations, double elapsed, const SearchTree& ts,
                          const SearchTree& tg) const {
    const std::vector<Cable>& used = !path.empty() ? path : raw_path;
    auto path_cost = [](const std::vector<Cable>& p) {
      double c = 0.0;
      for (std::size_t k = 0; k + 1 < p.size(); ++k) c += configuration_distance(p[k], p[k + 1]);
      return c;
    };
    const double cost = path_cost(used);
    const double raw_cost = path_cost(raw_path);
    double def_cost = 0.0;
    std::vector<double> edges;
    for (std::size_t k = 0; k + 1 < used.size(); ++k) {
      edges.push_back(configuration_distance(used[k], used[k + 1]));
      def_cost += deformation_energy(used[k + 1], used[k]);
    }
    const double max_edge = edges.empty() ? std::numeric_limits<double>::quiet_NaN()
                                          : *std::max_element(edges.begin(), edges.end());
    double min_clearance = std::numeric_limits<double>::quiet_NaN();
    if (!used.empty()) {
      min_clearance = std::numeric_limits<double>::infinity();
      for (const auto& c : used) {
        min_clearance = std::min(min_clearance, min_configuration_clearance(c, scenario_.obstacles, params_.segment_samples));
      }
    }
    const int validation_samples = std::max(params_.transition_samples, params_.shortcut_validation_samples);
    bool transition_ok = used.size() > 1;
    for (std::size_t k = 0; k + 1 < used.size() && transition_ok; ++k) {
      transition_ok = path_free(used[k], used[k + 1], scenario_.obstacles, validation_samples,
                                params_.robot_execution_feasibility, params_.robot_tcp_clearance,
                                robot_obstacle_clearance_value(), params_.robot_swept_tcp_clearance,
                                params_.max_cable_axis_turn_rad);
    }
    bool static_ok = !used.empty();
    for (const auto& c : used) {
      static_ok = static_ok && configuration_admissible(c, scenario_.obstacles, ell_min_, ell_max_, scenario_.workspace);
    }
    double max_seg_error = std::numeric_limits<double>::quiet_NaN();
    if (!used.empty()) {
      max_seg_error = 0.0;
      for (const auto& c : used) {
        for (double len : segment_lengths(c)) max_seg_error = std::max(max_seg_error, std::abs(len - ell0_));
      }
    }
    return {scenario_.name,
            success,
            timed_out,
            iterations,
            elapsed,
            params_.has_time_limit ? params_.scenario_time_limit_s : std::numeric_limits<double>::quiet_NaN(),
            static_cast<int>(ts.nodes.size()),
            static_cast<int>(tg.nodes.size()),
            static_cast<int>(raw_path.size()),
            static_cast<int>(path.size()),
            params_.shortcut_refinement,
            params_.endpoint_reorder,
            params_.robot_execution_feasibility,
            params_.robot_execution_feasibility ? params_.robot_tcp_clearance : 0.0,
            params_.robot_execution_feasibility
                ? (params_.has_robot_obstacle_clearance ? params_.robot_obstacle_clearance : params_.robot_tcp_clearance)
                : 0.0,
            params_.robot_execution_feasibility ? params_.robot_swept_tcp_clearance : 0.0,
            params_.preserve_terminal_order,
            endpoint_reversals_,
            params_.epsilon,
            max_edge,
            validation_samples,
            cost,
            raw_cost,
            raw_cost <= EPS ? 0.0 : 100.0 * (raw_cost - cost) / raw_cost,
            def_cost,
            min_clearance,
            max_seg_error,
            static_ok,
            transition_ok,
            reject_static_,
            reject_no_parent_,
            reject_no_connect_};
  }
};

struct DloSetup {
  Workspace workspace;
  std::vector<Polygon> polygons;
  std::vector<Circle> circles;
  Cable start;
  Cable goal;
};

DloSetup get_dlo_problem_setup() {
  DloSetup s;
  s.workspace = {0.0, 1.0, 0.0, 0.6};
  s.polygons = {
      {{0.22, 0.15}, {0.36, 0.17}, {0.38, 0.30}, {0.27, 0.35}, {0.19, 0.27}},
      {{0.60, 0.36}, {0.76, 0.33}, {0.82, 0.45}, {0.71, 0.53}, {0.58, 0.47}},
  };
  s.circles = {{{0.51, 0.20}, 0.075}, {{0.79, 0.18}, 0.055}};
  const int n_nodes = 8;
  const double arc_radius = 0.20;
  const double arc_span = deg2rad(80.0);
  s.start = circular_arc_nodes({0.310, 0.320}, arc_radius, deg2rad(130.0), deg2rad(130.0) - arc_span, n_nodes);
  s.goal = circular_arc_nodes({0.660, 0.230}, arc_radius, deg2rad(230.0), deg2rad(230.0) + arc_span, n_nodes);
  return s;
}

Cable dlo_line(const Vec2& center, double angle, double length, int n_nodes) {
  const Vec2 d{std::cos(angle), std::sin(angle)};
  return line_nodes(center - 0.5 * length * d, center + 0.5 * length * d, n_nodes);
}

std::vector<Scenario> build_gcr_last_benchmark_scenarios(int n_nodes) {
  Workspace workspace{0.0, 1.0, 0.0, 0.6};
  std::vector<Scenario> scenarios;
  scenarios.reserve(10);

  scenarios.push_back({"scenario_01_empty_rotation",
                       "01: Rotation with top-left + top-right obstacles",
                       "gcr_last_benchmark",
                       workspace,
                       {{rectangle(0.27, 0.38, 0.45, 0.55), rectangle(0.55, 0.38, 0.73, 0.55)}, {}},
                       straight_at({0.50, 0.30}, 0.0, n_nodes),
                       straight_at({0.50, 0.30}, PI / 2.0, n_nodes),
                       0.22,
                       101,
                       {{0.50, 0.30}, {0.50, 0.20}, {0.55, 0.15}, {0.50, 0.20}, {0.50, 0.30}},
                       {}});

  scenarios.push_back({"scenario_02_block_reorient",
                       "02: Reorient around an L-extended block with rounded target",
                       "gcr_last_benchmark",
                       workspace,
                       {{rectangle(0.30, 0.20, 0.45, 0.45), rectangle(0.10, 0.40, 0.30, 0.45)}, {}},
                       straight_at({0.15, 0.52}, 0.0, n_nodes),
                       rounded_l_down_right({0.18, 0.34}, n_nodes),
                       0.22,
                       202,
                       {{0.15, 0.52}, {0.05, 0.48}, {0.05, 0.32}, {0.10, 0.20}, {0.09, 0.31}},
                       {}});

  scenarios.push_back({"scenario_03_L_narrow_corridor",
                       "03: Long-horizontal corridor ('-----|')",
                       "gcr_last_benchmark",
                       workspace,
                       {{rectangle(0.00, 0.4025, 1.00, 0.60), rectangle(0.00, 0.00, 0.65, 0.25),
                         rectangle(0.80, 0.00, 1.00, 0.2475), rectangle(0.80, 0.2525, 1.00, 0.3975)},
                        {}},
                       straight_at({0.18, 0.325}, 0.0, n_nodes),
                       straight_at({0.725, 0.165}, PI / 2.0, n_nodes),
                       0.22,
                       303,
                       {{0.18, 0.325}, {0.32, 0.325}, {0.48, 0.325}, {0.64, 0.325}, {0.725, 0.30},
                        {0.725, 0.22}, {0.725, 0.165}},
                       {}});

  scenarios.push_back({"scenario_04_narrow_gate_mirror_L",
                       "04: 0.15 narrow gate with opposite-concavity arcs",
                       "gcr_last_benchmark",
                       workspace,
                       {{rectangle(0.35, 0.00, 0.65, 0.225), rectangle(0.35, 0.375, 0.65, 0.60)}, {}},
                       arc_at({0.10, 0.30}, -PI / 2.0, PI / 2.0, n_nodes),
                       arc_at({0.90, 0.30}, PI / 2.0, 3.0 * PI / 2.0, n_nodes),
                       0.22,
                       404,
                       {{0.15, 0.30}, {0.25, 0.30}, {0.40, 0.30}, {0.50, 0.30}, {0.60, 0.30},
                        {0.75, 0.30}, {0.85, 0.30}},
                       {}});

  scenarios.push_back({"scenario_05_arc_to_straight",
                       "05: Arc -> distant straight through zigzag",
                       "gcr_last_benchmark",
                       workspace,
                       {{},
                        {{{0.28, 0.42}, 0.035}, {{0.36, 0.18}, 0.035}, {{0.44, 0.42}, 0.035},
                         {{0.52, 0.18}, 0.035}, {{0.60, 0.42}, 0.035}, {{0.68, 0.18}, 0.035}}},
                       arc_at({0.18, 0.30}, -PI / 2.0, PI / 2.0, n_nodes),
                       straight_at({0.85, 0.30}, 0.0, n_nodes),
                       0.22,
                       505,
                       {{0.20, 0.30}, {0.32, 0.28}, {0.40, 0.30}, {0.48, 0.28}, {0.56, 0.30},
                        {0.64, 0.28}, {0.72, 0.30}, {0.80, 0.30}},
                       {}});

  scenarios.push_back({"scenario_06_cross_corridor",
                       "06: Cross corridor with TR-corner goal arc",
                       "gcr_last_benchmark",
                       workspace,
                       {{rectangle(0.00, 0.00, 0.38, 0.23), rectangle(0.52, 0.00, 1.00, 0.23),
                         rectangle(0.00, 0.37, 0.38, 0.60), rectangle(0.52, 0.37, 1.00, 0.60)},
                        {}},
                       straight_at({0.18, 0.30}, 0.0, n_nodes),
                       arc_at({0.46, 0.30}, 0.0, PI, n_nodes),
                       0.22,
                       606,
                       {{0.18, 0.30}, {0.30, 0.30}, {0.40, 0.30}, {0.45, 0.35}, {0.46, 0.32}},
                       {}});

  scenarios.push_back({"scenario_07_L_corridor_hard",
                       "07: L-corridor with two bend obstacles",
                       "gcr_last_benchmark",
                       workspace,
                       {{rectangle(0.00, 0.40, 1.00, 0.60), rectangle(0.00, 0.00, 0.40, 0.25),
                         rectangle(0.55, 0.00, 1.00, 0.25), rectangle(0.55, 0.25, 1.00, 0.40)},
                        {{{0.43, 0.36}, 0.022}, {{0.43, 0.28}, 0.022}}},
                       straight_at({0.18, 0.325}, 0.0, n_nodes),
                       straight_at({0.475, 0.195}, PI / 2.0, n_nodes),
                       0.22,
                       707,
                       {{0.18, 0.325}, {0.30, 0.325}, {0.40, 0.32}, {0.475, 0.30}, {0.475, 0.24},
                        {0.475, 0.195}},
                       {}});

  DloSetup dlo = get_dlo_problem_setup();
  const int dlo_n = static_cast<int>(dlo.start.size());
  const double dlo_len = cable_length(dlo.start);
  std::vector<Cable> dlo_guides = {
      dlo.start,
      dlo_line({0.30, 0.50}, 0.0, dlo_len, dlo_n),
      dlo_line({0.20, 0.50}, 0.0, dlo_len, dlo_n),
      dlo_line({0.135, 0.45}, -0.5 * PI, dlo_len, dlo_n),
      dlo_line({0.135, 0.36}, -0.5 * PI, dlo_len, dlo_n),
      dlo_line({0.135, 0.20}, -0.5 * PI, dlo_len, dlo_n),
      dlo_line({0.30, 0.07}, 0.0, dlo_len, dlo_n),
      dlo_line({0.52, 0.07}, 0.0, dlo_len, dlo_n),
      dlo.goal,
  };
  Vec2 dlo_start_mean{0.0, 0.0};
  Vec2 dlo_goal_mean{0.0, 0.0};
  for (const auto& p : dlo.start) dlo_start_mean += p;
  for (const auto& p : dlo.goal) dlo_goal_mean += p;
  dlo_start_mean = dlo_start_mean / static_cast<double>(dlo.start.size());
  dlo_goal_mean = dlo_goal_mean / static_cast<double>(dlo.goal.size());
  scenarios.push_back({"scenario_08_dlo_workspace",
                       "08: DLO workspace (generate_dlo_workspace_figure)",
                       "gcr_last_benchmark",
                       dlo.workspace,
                       {dlo.polygons, dlo.circles},
                       dlo.start,
                       dlo.goal,
                       0.30,
                       808,
                       {dlo_start_mean, {0.15, 0.47}, {0.12, 0.12}, {0.47, 0.07}, dlo_goal_mean},
                       dlo_guides});

  scenarios.push_back({"scenario_09_S_corridor",
                       "09: S-corridor, arcs with different curvatures",
                       "gcr_last_benchmark",
                       workspace,
                       {{rectangle(0.00, 0.00, 0.30, 0.22), rectangle(0.30, 0.30, 0.65, 0.60),
                         rectangle(0.65, 0.00, 1.00, 0.22)},
                        {}},
                       arc_at({0.15, 0.45}, PI, 0.0, n_nodes),
                       arc_at({0.80, 0.45}, -PI / 4.0, PI / 4.0, n_nodes),
                       0.22,
                       909,
                       {{0.15, 0.52}, {0.30, 0.26}, {0.50, 0.18}, {0.70, 0.26}, {0.85, 0.40}},
                       {}});

  scenarios.push_back({"scenario_10_cluttered_bent",
                       "10: Cluttered field, arcs with different curvatures",
                       "gcr_last_benchmark",
                       {0.0, 1.0, -0.05, 0.6},
                       {{},
                        {{{0.25, 0.20}, 0.035}, {{0.30, 0.45}, 0.035}, {{0.45, 0.30}, 0.035},
                         {{0.50, 0.10}, 0.035}, {{0.55, 0.50}, 0.035}, {{0.65, 0.22}, 0.035},
                         {{0.75, 0.45}, 0.035}, {{0.85, 0.18}, 0.035}}},
                       arc_at({0.10, 0.30}, -PI / 3.0, PI / 3.0, n_nodes),
                       arc_at({1.04, 0.10}, 5.0 * PI / 6.0, 7.0 * PI / 6.0, n_nodes),
                       0.22,
                       1010,
                       {{0.15, 0.30}, {0.30, 0.30}, {0.40, 0.40}, {0.55, 0.30}, {0.70, 0.25},
                        {0.82, 0.16}, {0.82, 0.10}},
                       {}});

  return scenarios;
}

std::vector<Scenario> build_current_selected_scenarios(int n_nodes) {
  Workspace workspace{0.0, 1.0, 0.0, 0.6};
  const Cable start_2 = arc_at({0.19, 0.12}, PI, 0.0, n_nodes);
  const Cable goal_2 = arc_at({0.81, 0.48}, 0.0, -PI, n_nodes);
  const Cable start_6 = line_nodes({0.07, 0.08}, {0.31, 0.08}, n_nodes);
  const Cable goal_6 = line_nodes({0.53, 0.43}, {0.77, 0.43}, n_nodes);
  const Cable start_8 = line_nodes({0.05, 0.53}, {0.32, 0.53}, n_nodes);
  const Cable goal_8 = line_nodes({0.68, 0.07}, {0.95, 0.07}, n_nodes);
  const double scenario_4_segment = 0.24 / static_cast<double>(n_nodes - 1);
  const Cable start_4 = arc_at({0.17, 0.10}, PI, 2.0 * PI, n_nodes, scenario_4_segment);
  const Cable goal_4 = line_nodes({0.95, 0.10}, {0.95, 0.34}, n_nodes);

  DloSetup current_dlo = get_dlo_problem_setup();
  if (!current_dlo.polygons.empty()) {
    for (auto& p : current_dlo.polygons[0]) p.x -= 0.10;
  }
  if (!current_dlo.circles.empty()) current_dlo.circles[0].center.x -= 0.10;
  if (current_dlo.circles.size() > 1) current_dlo.circles[1].center.x += 0.10;
  const int current_dlo_n = static_cast<int>(current_dlo.start.size());
  const double current_dlo_len = cable_length(current_dlo.start);
  std::vector<Cable> current_dlo_guides = {
      current_dlo.start,
      dlo_line({0.30, 0.50}, 0.0, current_dlo_len, current_dlo_n),
      dlo_line({0.20, 0.50}, 0.0, current_dlo_len, current_dlo_n),
      dlo_line({0.135, 0.45}, -0.5 * PI, current_dlo_len, current_dlo_n),
      dlo_line({0.135, 0.36}, -0.5 * PI, current_dlo_len, current_dlo_n),
      dlo_line({0.135, 0.20}, -0.5 * PI, current_dlo_len, current_dlo_n),
      dlo_line({0.30, 0.07}, 0.0, current_dlo_len, current_dlo_n),
      dlo_line({0.52, 0.07}, 0.0, current_dlo_len, current_dlo_n),
      current_dlo.goal,
  };
  Vec2 current_dlo_start_mean{0.0, 0.0};
  Vec2 current_dlo_goal_mean{0.0, 0.0};
  for (const auto& p : current_dlo.start) current_dlo_start_mean += p;
  for (const auto& p : current_dlo.goal) current_dlo_goal_mean += p;
  current_dlo_start_mean = current_dlo_start_mean / static_cast<double>(current_dlo.start.size());
  current_dlo_goal_mean = current_dlo_goal_mean / static_cast<double>(current_dlo.goal.size());

  return {
      {"scenario_2_narrow_gate",
       "Scenario 2: narrow gate between two blocks",
       "current_code_selected",
       workspace,
       {{rectangle(0.42, 0.00, 0.58, 0.24), rectangle(0.42, 0.36, 0.58, 0.60)}, {}},
       start_2,
       goal_2,
       0.205,
       23,
       {{0.19, 0.17}, {0.34, 0.18}, {0.50, 0.30}, {0.66, 0.42}, {0.81, 0.43}},
       {}},
      {"scenario_6_u_shaped_trap",
       "Scenario 6: U-shaped obstacle trap",
       "current_code_selected",
       workspace,
       {{rectangle(0.39, 0.28, 0.45, 0.56), rectangle(0.45, 0.50, 0.73, 0.56),
         rectangle(0.45, 0.28, 0.73, 0.34)},
        {}},
       goal_6,
       start_6,
       0.205,
       61,
       {{0.65, 0.43}, {0.78, 0.43}, {0.78, 0.18}, {0.34, 0.08}, {0.19, 0.08}},
       {}},
      {"scenario_8_two_shelves",
       "Scenario 8: two long shelf obstacles",
       "current_code_selected",
       workspace,
       {{rectangle(0.48, 0.39, 0.62, 0.43), rectangle(0.38, 0.17, 0.52, 0.21)}, {}},
       start_8,
       goal_8,
       0.360,
       83,
       {{0.185, 0.53}, {0.36, 0.50}, {0.46, 0.30}, {0.66, 0.22}, {0.82, 0.07}, {0.815, 0.07}},
       {}},
      {"scenario_4_staggered_blocks",
       "Scenario 4: staggered rectangular blocks",
       "current_code_selected",
       workspace,
       {{rectangle(0.24, 0.18, 0.36, 0.40), rectangle(0.48, 0.00, 0.60, 0.24),
         rectangle(0.74, 0.20, 0.86, 0.58)},
        {}},
       start_4,
       goal_4,
       0.190,
       41,
       {{0.17, 0.10}, {0.28, 0.12}, {0.42, 0.28}, {0.64, 0.28}, {0.90, 0.22}, {0.95, 0.22}},
       {}},
      {"scenario_9_dlo_workspace",
       "Scenario 9: DLO workspace from generate_dlo_workspace_figure.py",
       "current_code_selected",
       current_dlo.workspace,
       {current_dlo.polygons, current_dlo.circles},
       current_dlo.start,
       current_dlo.goal,
       0.300,
       107,
       {current_dlo_start_mean, {0.15, 0.47}, {0.12, 0.12}, {0.47, 0.07}, current_dlo_goal_mean},
       current_dlo_guides},
  };
}

std::vector<Scenario> build_final_scenarios(int n_nodes) {
  std::vector<Scenario> scenarios = build_gcr_last_benchmark_scenarios(n_nodes);
  std::vector<Scenario> selected = build_current_selected_scenarios(n_nodes);
  scenarios.insert(scenarios.end(), selected.begin(), selected.end());
  return scenarios;
}

void validate_scenarios(const std::vector<Scenario>& scenarios, const PlannerParams& params) {
  for (const auto& scenario : scenarios) {
    if (scenario.start.size() != scenario.goal.size()) {
      throw std::runtime_error(scenario.name + ": start and goal node counts differ");
    }
    const double ell0 = nominal_segment_length(scenario.start);
    const double ell_min = params.ell_min_ratio * ell0;
    const double ell_max = params.ell_max_ratio * ell0;
    if (!configuration_admissible(scenario.start, scenario.obstacles, ell_min, ell_max, scenario.workspace)) {
      throw std::runtime_error(scenario.name + ": start configuration is not collision-free");
    }
    if (!configuration_admissible(scenario.goal, scenario.obstacles, ell_min, ell_max, scenario.workspace)) {
      throw std::runtime_error(scenario.name + ": goal configuration is not collision-free");
    }
  }
}

std::string csv_value(double value) {
  if (std::isnan(value)) return "nan";
  if (std::isinf(value)) return value > 0 ? "inf" : "-inf";
  std::ostringstream ss;
  ss << std::setprecision(10) << value;
  return ss.str();
}

std::string csv_bool(bool value) { return value ? "true" : "false"; }

std::string csv_escape(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
  std::string out = "\"";
  for (char ch : value) {
    if (ch == '"') out += "\"\"";
    else out += ch;
  }
  out += "\"";
  return out;
}

void write_metrics_header(std::ofstream& out, bool include_run_columns) {
  out << "scenario,success,timed_out,iterations,time_s,time_limit_s,start_vertices,goal_vertices,"
         "raw_path_configs,refined_path_configs,shortcut_refinement_enabled,endpoint_reorder_enabled,"
         "robot_execution_feasibility_enabled,robot_tcp_clearance_m,robot_obstacle_clearance_m,"
         "robot_swept_tcp_clearance_m,preserve_terminal_order_enabled,"
         "endpoint_order_reversals,epsilon_m,max_edge_distance_m,"
         "shortcut_validation_samples,path_cost_J,raw_path_cost_J,shortcut_reduction_pct,deformation_energy,"
         "min_clearance_m,max_segment_length_error_m,static_collision_free,transition_path_free,reject_static,"
         "reject_no_parent,reject_no_connect";
  if (include_run_columns) out << ",final_scenario_number,source,run_index,seed";
  out << "\n";
}

void write_metrics_row(std::ofstream& out, const Metrics& m, int scenario_number = -1, const std::string& source = "",
                       int run_index = -1, int seed = 0) {
  out << m.scenario << "," << csv_bool(m.success) << "," << csv_bool(m.timed_out) << "," << m.iterations << ","
      << csv_value(m.time_s) << "," << csv_value(m.time_limit_s) << "," << m.start_vertices << "," << m.goal_vertices
      << "," << m.raw_path_configs << "," << m.refined_path_configs << "," << csv_bool(m.shortcut_refinement_enabled)
      << "," << csv_bool(m.endpoint_reorder_enabled) << "," << csv_bool(m.robot_execution_feasibility_enabled)
      << "," << csv_value(m.robot_tcp_clearance_m) << "," << csv_value(m.robot_obstacle_clearance_m)
      << "," << csv_value(m.robot_swept_tcp_clearance_m) << "," << csv_bool(m.preserve_terminal_order_enabled)
      << "," << m.endpoint_order_reversals
      << "," << csv_value(m.epsilon_m) << "," << csv_value(m.max_edge_distance_m) << ","
      << m.shortcut_validation_samples << "," << csv_value(m.path_cost_J) << "," << csv_value(m.raw_path_cost_J)
      << "," << csv_value(m.shortcut_reduction_pct) << "," << csv_value(m.deformation_energy) << ","
      << csv_value(m.min_clearance_m) << "," << csv_value(m.max_segment_length_error_m) << ","
      << csv_bool(m.static_collision_free) << "," << csv_bool(m.transition_path_free) << "," << m.reject_static << ","
      << m.reject_no_parent << "," << m.reject_no_connect;
  if (scenario_number > 0) out << "," << scenario_number << "," << source << "," << run_index << "," << seed;
  out << "\n";
}

void write_solution_path_csv(const fs::path& path, const PlanResult& result, int scenario_number, int run_index, int seed) {
  const std::vector<Cable>& solution = !result.path.empty() ? result.path : result.raw_path;
  std::ofstream out(path);
  out << "# scenario_number: " << scenario_number << "\n";
  out << "# scenario: " << result.scenario.name << "\n";
  out << "# title: " << result.scenario.title << "\n";
  out << "# source: " << result.scenario.source << "\n";
  out << "# run_index: " << run_index << "\n";
  out << "# seed: " << seed << "\n";
  out << "# success: " << csv_bool(result.success) << "\n";
  const bool shortcut_enabled = result.metrics.shortcut_refinement_enabled;
  out << "# path_type: " << (shortcut_enabled ? "shortcut_refined" : "raw_connected_no_shortcut") << "\n";
  out << "# endpoint_reorder: " << csv_bool(result.metrics.endpoint_reorder_enabled) << "\n";
  out << "# robot_execution_feasibility: " << csv_bool(result.metrics.robot_execution_feasibility_enabled) << "\n";
  out << "# robot_tcp_clearance_m: " << csv_value(result.metrics.robot_tcp_clearance_m) << "\n";
  out << "# robot_obstacle_clearance_m: " << csv_value(result.metrics.robot_obstacle_clearance_m) << "\n";
  out << "# robot_swept_tcp_clearance_m: " << csv_value(result.metrics.robot_swept_tcp_clearance_m) << "\n";
  out << "# preserve_terminal_order: " << csv_bool(result.metrics.preserve_terminal_order_enabled) << "\n";
  out << "# endpoint_order_reversals: " << result.metrics.endpoint_order_reversals << "\n";
  out << "# configs: " << solution.size() << "\n";
  out << "# nodes: " << (solution.empty() ? 0 : solution.front().size()) << "\n";
  out << "config_index,node_index,x,y\n";
  for (std::size_t c = 0; c < solution.size(); ++c) {
    for (std::size_t n = 0; n < solution[c].size(); ++n) {
      out << c << "," << n << "," << csv_value(solution[c][n].x) << "," << csv_value(solution[c][n].y) << "\n";
    }
  }
}

struct Args {
  fs::path output_dir = fs::path("final_results/cpp_benchmark");
  std::vector<int> scenarios;
  int nodes = 10;
  int runs = 1;
  int max_iterations = 1600;
  double scenario_time_limit = 120.0;
  int seed_offset = 0;
  bool shortcut_refinement = true;
  bool endpoint_reorder = false;
  bool robot_execution_feasibility = false;
  double robot_tcp_clearance = 0.0;
  bool has_robot_obstacle_clearance = false;
  double robot_obstacle_clearance = 0.0;
  double robot_swept_tcp_clearance = 0.0;
  bool preserve_terminal_order = true;
  double endpoint_reorder_cost_ratio = 0.75;
  double endpoint_reorder_tangent_margin = 0.05;
  bool robot_base_aware_endpoint_reorder = false;
  bool robot_base_vector_endpoint_reorder = false;
  bool robot_tcp_vector_sector_reorder = false;
  bool robot_tcp_vector_sector_feasibility = false;
  double robot_base_vector_max_turn_deg = 100.0;
  double robot_base_vector_margin_deg = 5.0;
  double robot_tcp_vector_sector_margin_deg = 5.0;
  double max_cable_axis_turn_deg = 180.0;
  double robot_base_arm_clearance = 0.0;
  int robot_base_transition_samples = 8;
  bool save_all_success_paths = false;
  bool swap_start_goal = false;
};

Args parse_args(int argc, char** argv) {
  Args args;
  bool output_set = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto need_value = [&](const std::string& name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(name + " requires a value");
      return argv[++i];
    };
    if (a == "--scenario") {
      args.scenarios.push_back(std::stoi(need_value(a)));
    } else if (a == "--nodes") {
      args.nodes = std::stoi(need_value(a));
    } else if (a == "--runs") {
      args.runs = std::stoi(need_value(a));
    } else if (a == "--max-iterations") {
      args.max_iterations = std::stoi(need_value(a));
    } else if (a == "--scenario-time-limit") {
      args.scenario_time_limit = std::stod(need_value(a));
    } else if (a == "--seed-offset") {
      args.seed_offset = std::stoi(need_value(a));
    } else if (a == "--no-shortcut-refinement") {
      args.shortcut_refinement = false;
    } else if (a == "--endpoint-reorder") {
      args.endpoint_reorder = true;
    } else if (a == "--endpoint-reorder-cost-ratio") {
      args.endpoint_reorder_cost_ratio = std::stod(need_value(a));
    } else if (a == "--endpoint-reorder-tangent-margin") {
      args.endpoint_reorder_tangent_margin = std::stod(need_value(a));
    } else if (a == "--robot-base-aware-endpoint-reorder") {
      args.robot_base_aware_endpoint_reorder = true;
    } else if (a == "--robot-base-vector-endpoint-reorder") {
      args.robot_base_vector_endpoint_reorder = true;
    } else if (a == "--robot-tcp-vector-sector-reorder") {
      args.robot_tcp_vector_sector_reorder = true;
    } else if (a == "--robot-tcp-vector-sector-feasibility") {
      args.robot_tcp_vector_sector_feasibility = true;
    } else if (a == "--robot-base-vector-max-turn-deg") {
      args.robot_base_vector_max_turn_deg = std::stod(need_value(a));
    } else if (a == "--robot-base-vector-margin-deg") {
      args.robot_base_vector_margin_deg = std::stod(need_value(a));
    } else if (a == "--robot-tcp-vector-sector-margin-deg") {
      args.robot_tcp_vector_sector_margin_deg = std::stod(need_value(a));
    } else if (a == "--max-cable-axis-turn-deg") {
      args.max_cable_axis_turn_deg = std::stod(need_value(a));
    } else if (a == "--robot-base-arm-clearance") {
      args.robot_base_arm_clearance = std::stod(need_value(a));
    } else if (a == "--robot-base-transition-samples") {
      args.robot_base_transition_samples = std::stoi(need_value(a));
    } else if (a == "--robot-execution-feasibility") {
      args.robot_execution_feasibility = true;
    } else if (a == "--robot-tcp-clearance") {
      args.robot_tcp_clearance = std::stod(need_value(a));
    } else if (a == "--robot-obstacle-clearance") {
      args.robot_obstacle_clearance = std::stod(need_value(a));
      args.has_robot_obstacle_clearance = true;
    } else if (a == "--robot-swept-tcp-clearance") {
      args.robot_swept_tcp_clearance = std::stod(need_value(a));
    } else if (a == "--allow-terminal-reversal") {
      args.preserve_terminal_order = false;
    } else if (a == "--save-all-success-paths") {
      args.save_all_success_paths = true;
    } else if (a == "--swap-start-goal") {
      args.swap_start_goal = true;
    } else if (a == "--help" || a == "-h") {
      std::cout << "Usage: gcr_benchmark [output_dir] [--scenario N] [--runs N] [--nodes N]\n"
                   "                     [--max-iterations N] [--scenario-time-limit S]\n"
                   "                     [--seed-offset N] [--no-shortcut-refinement] [--endpoint-reorder]\n"
                   "                     [--endpoint-reorder-cost-ratio R]\n"
                   "                     [--endpoint-reorder-tangent-margin M]\n"
                    "                     [--robot-base-aware-endpoint-reorder]\n"
                    "                     [--robot-base-vector-endpoint-reorder]\n"
                    "                     [--robot-tcp-vector-sector-reorder]\n"
                    "                     [--robot-tcp-vector-sector-feasibility]\n"
                    "                     [--robot-base-vector-max-turn-deg D]\n"
                    "                     [--robot-base-vector-margin-deg D]\n"
                    "                     [--robot-tcp-vector-sector-margin-deg D]\n"
                    "                     [--max-cable-axis-turn-deg D]\n"
                   "                     [--robot-base-arm-clearance M]\n"
                   "                     [--robot-base-transition-samples N]\n"
                   "                     [--robot-execution-feasibility] [--robot-tcp-clearance M]\n"
                   "                     [--robot-obstacle-clearance M] [--robot-swept-tcp-clearance M]\n"
                   "                     [--allow-terminal-reversal]\n"
                   "                     [--swap-start-goal]\n"
                   "                     [--save-all-success-paths]\n";
      std::exit(0);
    } else if (!a.empty() && a[0] == '-') {
      throw std::runtime_error("unknown option: " + a);
    } else if (!output_set) {
      args.output_dir = a;
      output_set = true;
    } else {
      throw std::runtime_error("unexpected positional argument: " + a);
    }
  }
  if (args.runs < 1) throw std::runtime_error("--runs must be at least 1");
  if (args.robot_tcp_clearance < 0.0) throw std::runtime_error("--robot-tcp-clearance must be non-negative");
  if (args.has_robot_obstacle_clearance && args.robot_obstacle_clearance < 0.0) {
    throw std::runtime_error("--robot-obstacle-clearance must be non-negative");
  }
  if (args.robot_swept_tcp_clearance < 0.0) {
    throw std::runtime_error("--robot-swept-tcp-clearance must be non-negative");
  }
  if (args.endpoint_reorder_cost_ratio <= 0.0) {
    throw std::runtime_error("--endpoint-reorder-cost-ratio must be positive");
  }
  if (args.robot_base_arm_clearance < 0.0) throw std::runtime_error("--robot-base-arm-clearance must be non-negative");
  if (args.robot_base_transition_samples < 1) {
    throw std::runtime_error("--robot-base-transition-samples must be at least 1");
  }
  if (args.robot_base_vector_max_turn_deg < 0.0 || args.robot_base_vector_max_turn_deg > 180.0) {
    throw std::runtime_error("--robot-base-vector-max-turn-deg must be in [0, 180]");
  }
  if (args.robot_base_vector_margin_deg < 0.0) {
    throw std::runtime_error("--robot-base-vector-margin-deg must be non-negative");
  }
  if (args.robot_tcp_vector_sector_margin_deg < 0.0) {
    throw std::runtime_error("--robot-tcp-vector-sector-margin-deg must be non-negative");
  }
  if (args.max_cable_axis_turn_deg < 0.0 || args.max_cable_axis_turn_deg > 180.0) {
    throw std::runtime_error("--max-cable-axis-turn-deg must be in [0, 180]");
  }
  return args;
}

std::string scenario_dir_name(int number, const std::string& name) {
  std::ostringstream ss;
  ss << std::setw(2) << std::setfill('0') << number << "_" << name;
  return ss.str();
}

void swap_start_goal(Scenario& scenario) {
  std::swap(scenario.start, scenario.goal);
  std::reverse(scenario.guide_points.begin(), scenario.guide_points.end());
  std::reverse(scenario.guide_configs.begin(), scenario.guide_configs.end());
  scenario.title += " (start/goal swapped)";
  scenario.source += "_swapped";
}

struct SummaryRow {
  std::string scenario;
  std::string title;
  std::string source;
  int n_runs = 0;
  int n_success = 0;
  double success_rate = 0.0;
  double time_mean_s = 0.0;
  double time_std_s = 0.0;
  double iterations_mean = 0.0;
  double success_path_cost_mean = std::numeric_limits<double>::quiet_NaN();
};

double mean(const std::vector<double>& v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

double stddev(const std::vector<double>& v) {
  if (v.empty()) return std::numeric_limits<double>::quiet_NaN();
  const double m = mean(v);
  double accum = 0.0;
  for (double x : v) accum += (x - m) * (x - m);
  return std::sqrt(accum / static_cast<double>(v.size()));
}

SummaryRow summarize_runs(const Scenario& scenario, const std::vector<PlanResult>& results) {
  std::vector<double> times;
  std::vector<double> iterations;
  std::vector<double> success_costs;
  int n_success = 0;
  for (const auto& result : results) {
    times.push_back(result.elapsed_s);
    iterations.push_back(static_cast<double>(result.iterations));
    if (result.success) {
      ++n_success;
      success_costs.push_back(result.metrics.path_cost_J);
    }
  }
  return {scenario.name,
          scenario.title,
          scenario.source,
          static_cast<int>(results.size()),
          n_success,
          results.empty() ? 0.0 : static_cast<double>(n_success) / static_cast<double>(results.size()),
          mean(times),
          stddev(times),
          mean(iterations),
          mean(success_costs)};
}

}  // namespace gcr

int main(int argc, char** argv) {
  using namespace gcr;
  try {
    const Args args = parse_args(argc, argv);
    PlannerParams params;
    params.max_iterations = args.max_iterations;
    params.has_time_limit = args.scenario_time_limit > 0.0;
    params.scenario_time_limit_s = args.scenario_time_limit;
    params.shortcut_refinement = args.shortcut_refinement;
    params.endpoint_reorder = args.endpoint_reorder;
    params.robot_execution_feasibility = args.robot_execution_feasibility;
    params.robot_tcp_clearance = args.robot_tcp_clearance;
    params.has_robot_obstacle_clearance = args.has_robot_obstacle_clearance;
    params.robot_obstacle_clearance = args.robot_obstacle_clearance;
    params.robot_swept_tcp_clearance = args.robot_swept_tcp_clearance;
    params.preserve_terminal_order = args.preserve_terminal_order;
    params.endpoint_reorder_cost_ratio = args.endpoint_reorder_cost_ratio;
    params.endpoint_reorder_tangent_margin = args.endpoint_reorder_tangent_margin;
    params.robot_base_aware_endpoint_reorder = args.robot_base_aware_endpoint_reorder;
    params.robot_base_vector_endpoint_reorder = args.robot_base_vector_endpoint_reorder;
    params.robot_tcp_vector_sector_reorder = args.robot_tcp_vector_sector_reorder;
    params.robot_tcp_vector_sector_feasibility = args.robot_tcp_vector_sector_feasibility;
    params.robot_base_vector_max_turn_rad = deg2rad(args.robot_base_vector_max_turn_deg);
    params.robot_base_vector_margin_rad = deg2rad(args.robot_base_vector_margin_deg);
    params.robot_tcp_vector_sector_margin_rad = deg2rad(args.robot_tcp_vector_sector_margin_deg);
    params.max_cable_axis_turn_rad = deg2rad(args.max_cable_axis_turn_deg);
    params.robot_base_arm_clearance = args.robot_base_arm_clearance;
    params.robot_base_transition_samples = args.robot_base_transition_samples;

    std::vector<Scenario> scenarios = build_final_scenarios(args.nodes);
    if (args.swap_start_goal) {
      for (auto& scenario : scenarios) swap_start_goal(scenario);
    }
    for (auto& scenario : scenarios) scenario.seed += args.seed_offset;
    validate_scenarios(scenarios, params);

    std::vector<int> selected;
    if (args.scenarios.empty()) {
      selected.resize(scenarios.size());
      std::iota(selected.begin(), selected.end(), 1);
    } else {
      selected = args.scenarios;
    }
    for (int number : selected) {
      if (number < 1 || number > static_cast<int>(scenarios.size())) {
        throw std::runtime_error("--scenario must be in 1.." + std::to_string(scenarios.size()));
      }
    }

    fs::create_directories(args.output_dir);
    std::ofstream all_metrics(args.output_dir / "final_metrics_all.csv");
    write_metrics_header(all_metrics, true);

    std::cout << "Running " << selected.size() << " final scenario(s), runs=" << args.runs
              << ", shortcut_refinement=" << (params.shortcut_refinement ? "true" : "false")
              << ", endpoint_reorder=" << (params.endpoint_reorder ? "true" : "false")
              << ", robot_execution_feasibility=" << (params.robot_execution_feasibility ? "true" : "false")
              << ", robot_tcp_clearance=" << params.robot_tcp_clearance
              << ", robot_obstacle_clearance="
              << (params.has_robot_obstacle_clearance ? params.robot_obstacle_clearance : params.robot_tcp_clearance)
              << ", robot_swept_tcp_clearance=" << params.robot_swept_tcp_clearance
              << ", preserve_terminal_order=" << (params.preserve_terminal_order ? "true" : "false")
              << ", endpoint_reorder_cost_ratio=" << params.endpoint_reorder_cost_ratio
              << ", endpoint_reorder_tangent_margin=" << params.endpoint_reorder_tangent_margin
              << ", robot_base_aware_endpoint_reorder="
              << (params.robot_base_aware_endpoint_reorder ? "true" : "false")
              << ", robot_base_vector_endpoint_reorder="
              << (params.robot_base_vector_endpoint_reorder ? "true" : "false")
              << ", robot_tcp_vector_sector_reorder="
              << (params.robot_tcp_vector_sector_reorder ? "true" : "false")
              << ", robot_tcp_vector_sector_feasibility="
              << (params.robot_tcp_vector_sector_feasibility ? "true" : "false")
              << ", robot_base_vector_max_turn_deg=" << args.robot_base_vector_max_turn_deg
              << ", robot_base_vector_margin_deg=" << args.robot_base_vector_margin_deg
              << ", robot_tcp_vector_sector_margin_deg=" << args.robot_tcp_vector_sector_margin_deg
              << ", max_cable_axis_turn_deg=" << args.max_cable_axis_turn_deg
              << ", robot_base_arm_clearance=" << params.robot_base_arm_clearance
              << ", robot_base_transition_samples=" << params.robot_base_transition_samples
              << ", swap_start_goal=" << (args.swap_start_goal ? "true" : "false")
              << ", save_all_success_paths=" << (args.save_all_success_paths ? "true" : "false") << "\n";
    const auto total_start = std::chrono::steady_clock::now();
    std::vector<SummaryRow> summaries;

    for (int number : selected) {
      Scenario scenario = scenarios[static_cast<std::size_t>(number - 1)];
      const fs::path scenario_dir = args.output_dir / scenario_dir_name(number, scenario.name);
      fs::create_directories(scenario_dir);
      const fs::path run_paths_dir = scenario_dir / "run_paths";
      if (args.save_all_success_paths) fs::create_directories(run_paths_dir);
      std::ofstream scenario_metrics(scenario_dir / "metrics.csv");
      write_metrics_header(scenario_metrics, true);

      std::cout << "\n=== " << number << ": " << scenario.title << " [" << scenario.source << "] ===\n";
      std::vector<PlanResult> results;
      bool wrote_first_success_path = false;
      for (int run_idx = 0; run_idx < args.runs; ++run_idx) {
        Scenario run_scenario = scenario;
        run_scenario.seed = scenario.seed + run_idx * 1009;
        GCRPlanner planner(run_scenario, params);
        PlanResult result = planner.plan();
        const std::string status = result.success ? "success" : (result.timed_out ? "timeout" : "failure");
        std::cout << "  run " << (run_idx + 1) << "/" << args.runs << ": " << status
                  << ", iters=" << result.iterations << ", configs=" << result.metrics.refined_path_configs
                  << ", t=" << std::fixed << std::setprecision(2) << result.elapsed_s << "s\n";
        write_metrics_row(all_metrics, result.metrics, number, scenario.source, run_idx, run_scenario.seed);
        write_metrics_row(scenario_metrics, result.metrics, number, scenario.source, run_idx, run_scenario.seed);
        if (result.success && !wrote_first_success_path) {
          write_solution_path_csv(scenario_dir / "first_success_path.csv", result, number, run_idx, run_scenario.seed);
          wrote_first_success_path = true;
        }
        if (result.success && args.save_all_success_paths) {
          std::ostringstream stem;
          stem << "run_" << std::setw(4) << std::setfill('0') << run_idx << "_success_path.csv";
          write_solution_path_csv(run_paths_dir / stem.str(), result, number, run_idx, run_scenario.seed);
        }
        results.push_back(std::move(result));
      }
      summaries.push_back(summarize_runs(scenario, results));
    }

    std::ofstream summary(args.output_dir / "final_benchmark_summary.csv");
    summary << "scenario,title,source,n_runs,n_success,success_rate,time_mean_s,time_std_s,iterations_mean,"
               "success_path_cost_mean\n";
    for (const auto& row : summaries) {
      summary << csv_escape(row.scenario) << "," << csv_escape(row.title) << "," << csv_escape(row.source) << ","
              << row.n_runs << "," << row.n_success << "," << csv_value(row.success_rate) << ","
              << csv_value(row.time_mean_s) << ","
              << csv_value(row.time_std_s) << "," << csv_value(row.iterations_mean) << ","
              << csv_value(row.success_path_cost_mean) << "\n";
    }

    const double total_time = std::chrono::duration<double>(std::chrono::steady_clock::now() - total_start).count();
    std::cout << "\nDone in " << std::fixed << std::setprecision(2) << total_time << "s. Results: "
              << fs::absolute(args.output_dir).string() << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << "\n";
    return 1;
  }
}
