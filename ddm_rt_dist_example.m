% script demonstrating the use of ddm_rt_dist.m and ddm_rt_dist_full.m

figure; hold on;
set(gca, 'defaultlinelinewidth', 1);

% time discretisation, maximum time
delta_t = 1e-3;
t_max = 2.5;
ts = (1:ceil(t_max / delta_t)) * delta_t;

% constant bound, constant drift
% same as [g1 g2] = ddm_rt_dist_full(1, 1, -1, 1, 0, 0, delta_t, t_max);
[g1 g2] = ddm_rt_dist(1, 1, delta_t, t_max);
plot(ts, g1, 'k-');
plot(ts, -g2, 'k-');

% linear time-varying bound, constant drift
bound = 1 - 0.25 * ts;
% same as [g1 g2] = ddm_rt_dist_full(1, 1, -bound, bound, 0.25, -0.25, ...
%                                    delta_t, t_max);
[g1 g2] = ddm_rt_dist(1, bound, delta_t, t_max);
plot(ts, g1, 'b-');
plot(ts, -g2, 'b-');

% constant bound, time-varying drift
mu = 1 + 0.5 * ts;
% same as [g1 g2] = ddm_rt_dist_full(mu, 1, -1, 1, 0, 0, delta_t, t_max);
[g1 g2] = ddm_rt_dist(mu, 1, delta_t, t_max);
plot(ts, g1, 'r-');
plot(ts, -g2, 'r-');

% time-varying bound, time-varying drift
% same as [g1 g2] = ddm_rt_dist_full(mu, 1, -bound, bound, 0.25, -0.25, ...
%                                    delta_t, t_max);
[g1 g2] = ddm_rt_dist(mu, bound, delta_t, t_max);
plot(ts, g1, 'g-');
plot(ts, -g2, 'g-');

% constand bound, constant drift, leaky integration
% we need to use ddm_rt_dist_full, as ddm_rt_dist does not support leak
[g1 g2] = ddm_rt_dist_full(1, 1, -1, 1, 0, 0, delta_t, t_max, 0.5);
plot(ts, g1, 'b--');
plot(ts, -g2, 'b--');


% constant bound, constant drift, weighted integration
w = 0.5 * (1 + cos(ts * 2 * pi));
% same as [g1 g2] = ddm_rt_dist_full(w, w.^2, -1, 1, 0, 0, delta_t, t_max);
[g1 g2] = ddm_rt_dist(w, 1, delta_t, t_max, 1);
plot(ts, g1, 'r--');
plot(ts, -g2, 'r--');

% complete plot
xlabel('t');
ylabel('p(t)');
set(gca,'TickDir','out');
plot(get(gca, 'XLim'), [0 0], 'k--');
