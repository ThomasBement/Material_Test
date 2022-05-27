

clear;
close all;
addpath cmap;

fn = 'Master.xlsx';


opts = detectImportOptions(fn, 'VariableNamesRange', 'A1');
opts.DataRange = 'A2';

tab0 = readtable(fn, opts);

code = tab0.SampleCode; code = code(3:end); % material codes
treatment = tab0.Treatment; treatment = treatment(3:end); % material codes
name = tab0.Description; name = name(3:end); % material names
p_drop = tab0.PressureDrop_Pa_; p_drop = p_drop(3:end); % pressure drop
weight = tab0.Weight_g_m2_; weight = weight(3:end);
layers = tab0.Layers; layers = layers(3:end);

pen_col = find(strcmp(tab0.Properties.VariableNames,'x1'));
pen0 = table2array(tab0(3:end, pen_col:(pen_col+13))); % size-resolved penetrations
sizes = table2array(tab0(1, pen_col:(pen_col+13))); % aerodynamic diameter

A0 = pen0(:, 9); % channel 9 penetration
% A0 = -log(pen0(:, 9)) .* 1000 ./  p_drop;


% Find entries for each kind of code.
n_code = length(code); % number of material tests
f_W = false(n_code, 1); % flag for W (weaved) materials
f_K = false(n_code, 1); % flag for K (knitted) materials
f_CP = false(n_code, 1); % flag for CP materials
f_nW = false(n_code, 1); % flag for nW materials
f_other = false(n_code, 1); % flag for other materials
for ii=1:n_code
    if and(~isempty(code{ii}),1)%sisempty(treatment{ii}))
        f_W(ii) = strcmp(code{ii}(1),'W');
        f_K(ii) = strcmp(code{ii}(1),'K');
        f_CP(ii) = strcmp(code{ii}(1:2),'CP');
        f_nW(ii) = strcmp(code{ii}(1:2),'nW');
        if ~any([f_W(ii), f_K(ii), f_CP(ii), f_nW(ii)]); f_other(ii) = 1; end
    end
end



%== FIG 1: PENETRATION VS PRESSURE DROP ======================%
h = figure(1);
set(gca, 'FontName', 'Segoe UI Semilight')

% plot quality isolines
hold on;
delP_vec = 0:0.01:100;
qual_fun = @(QF) exp(-QF .* delP_vec ./ 1000);
QF_vec = [0.1,0.2,0.5,1,2,5,10,20,30,50,100,200,500];
for ii=1:length(QF_vec)
    if ii==2; hold on; end
    plot(delP_vec, 1-qual_fun(QF_vec(ii)), 'Color', [0.7,0.7,0.7]);
end
hold off;

hold on;
text(p_drop, 1-A0, code, 'FontSize', 8);
scatter(p_drop(f_W), 1-A0(f_W), weight(f_W)./3, 'filled', 'MarkerFaceColor', [254,45,76]./255, 'MarkerEdgeColor', [0,0,0]);
scatter(p_drop(f_K), 1-A0(f_K), weight(f_K)./3, 'filled', 'MarkerFaceColor', [44,60,176]./255, 'MarkerEdgeColor', [0,0,0]);
scatter(p_drop(f_CP), 1-A0(f_CP), weight(f_CP)./3, 'filled', 'MarkerFaceColor', [70,221,176]./255, 'MarkerEdgeColor', [0,0,0]);
scatter(p_drop(f_nW), 1-A0(f_nW), weight(f_nW)./3, 'filled', 'MarkerFaceColor', [254,231,31]./255, 'MarkerEdgeColor', [0,0,0]);
scatter(p_drop(f_other), 1-A0(f_other), weight(f_other)./3, 'filled', 'MarkerFaceColor', [1,1,1], 'MarkerEdgeColor', [0,0,0]);
hold off;

% add "limit of breathability"
hold on;
plot([85,85], [0,1], '--k');
hold off;

% add Zhao et al. data
Zhao = load('Zhao.mat');
hold on;
scatter(Zhao.p_drop, Zhao.pen/100, Zhao.weight./3, 'filled');
hold off;

ylim([0,1]);
xlim([0,90]);
set(gcf, 'Position', [20,60,1000,800]);
h.Renderer = 'Painters';


%{
%== FIG 2: PENETRATION (CORR.) VS PARTICLE SIZE ==================%
figure(2);
clf;
pen_corr = pen0.^(30./p_drop);
hold on;
f_text = any([f_W, f_K, f_CP, f_nW, f_other]')';
text(repmat(sizes(end),[sum(f_text),1]), ...
    1-pen_corr(f_text, end), code(f_text), 'FontSize', 8);
plot(sizes, 1-pen_corr(f_W, :), 'r.-');
plot(sizes, 1-pen_corr(f_K, :), 'b.-');
plot(sizes, 1-pen_corr(f_CP, :), 'g.-');
plot(sizes, 1-pen_corr(f_nW, :), 'y.-');
plot(sizes, 1-pen_corr(f_other, :), 'c.-');
hold off;



%== FIG 3: SELECT PENETRATION (CORR.) VS PARTICLE SIZE ============%
idxa = 1:length(code);
idxb = [];
select = {'N95', 'W15', 'W10', 'W3', 'nW5', 'K3', 'K4', 'K11', 'K7', ...
    'CP1', 'CP4', 'ASTM2', 'NMM'};
for ii=1:length(select) % loop through and find strings
    idxt = strcmp(code, select{ii}); idxt = idxa(idxt);
    idxb = [idxb, idxt];
end

idxc = [];
idxt = strcmp(code, 'nW2'); idxt = idxa(idxt); idxc = [idxc, idxt];
idxt = strcmp(code, 'nW3'); idxt = idxa(idxt); idxc = [idxc, idxt];
idxt = strcmp(code, 'nW4'); idxt = idxa(idxt); idxc = [idxc, idxt];

figure(3);
area(sizes, [min(1-pen_corr(idxc, :)); ...
    max(1-pen_corr(idxc, :)) - min(1-pen_corr(idxc, :))]');
hold on;
plot(sizes, 1-pen_corr(idxb, :)', '.-');
hold off;
text(repmat(sizes(end),[size(pen0(idxb,:),1),1]), ...
    1-pen_corr(idxb, end), code(idxb), 'FontSize', 8);
set(gca, 'XScale', 'log');
ylim([0,1.1]);




%== FIG 4: CHANGE IN PENETRATION (CORR.) VS PARTICLE SIZE ========%
idx_ml = find(layers>1); % multiple layer amterials
code(idx_ml);
idxd{1} = [28:30];
idxd{2} = [34,35];
idxd{3} = [50,52];
idxd{4} = [92,99];

figure(4);
plot(sizes, 1-pen_corr(idxd{1}, :)', '.-r');
hold on;
plot(sizes, 1-pen_corr(idxd{2}, :)', '.-g');
plot(sizes, 1-pen_corr(idxd{3}, :)', '.-b');
plot(sizes, 1-pen_corr(idxd{4}, :)', '.-y');
text(repmat(sizes(end),[size(pen0([idxd{:}],:),1),1]), ...
    1-pen_corr([idxd{:}], end), code([idxd{:}]), 'FontSize', 8);
hold off;

set(gca, 'XScale', 'log');
ylim([0,1.1]);



%== FIG 5: CHANGE IN PENETRATION (CORR.) VS PARTICLE SIZE ========%
idxC = idxa(strcmp(code, 'Combination C'));
comboCa = {'W5', 'W5', 'nW5'};
idxCa = [];
for ii=1:length(comboCa) % loop through and find strings
    idxt = strcmp(code, comboCa{ii}); idxt = idxa(idxt);
    idxCa = [idxCa, idxt(1)];
end

idxB = idxa(strcmp(code, 'Combination B'));
comboBa = {'nW3 x 2', 'W9', 'K7'};
idxBa = [];
for ii=1:length(comboBa) % loop through and find strings
    idxt = strcmp(code, comboBa{ii}); idxt = idxa(idxt);
    idxBa = [idxBa, idxt(1)];
end

comboDa = {'CP2', 'W9', 'K7'};
idxDa = [];
for ii=1:length(comboBa) % loop through and find strings
    idxt = strcmp(code, comboDa{ii}); idxt = idxa(idxt);
    idxDa = [idxDa, idxt(1)];
end

comboEa = {'nW5', 'W9', 'K7'};
idxEa = [];
for ii=1:length(comboBa) % loop through and find strings
    idxt = strcmp(code, comboEa{ii}); idxt = idxa(idxt);
    idxEa = [idxEa, idxt(1)];
end

figure(5);
plot(sizes, 1-prod(pen_corr(idxCa, :))', '.-');
hold on;
plot(sizes, 1-pen_corr(idxC, :)', '.-');
plot(sizes, 1-prod(pen_corr(idxBa, :))', '.-');
plot(sizes, 1-pen_corr(idxB, :)', '.-');
plot(sizes, 1-prod(pen_corr(idxDa, :))', '.-');
plot(sizes, 1-prod(pen_corr(idxEa, :))', '.-');
hold off;
set(gca, 'XScale', 'log');
ylim([0,1.1]);



%{
% generate a penetration curve
pen_n95 = pen0(find(strcmp('N95',code)), :); % 3M 1860 N95 mask
pen_nmm = pen0(find(strcmp('NMM',code)), :); % non-medical mask
pen_h400 = pen0(find(strcmp('L1 H400',name)), :); % L1 H400
pen_surg = pen0(find(strcmp('33 New ASTM Level 2 Primaguard surgical mask',name))-2, :); % surgical mask
pen_k7 = pen0(find(strcmp('K7 x 3',code)), :); % C6, knit cotton
pen_dbw = pen0(find(strcmp('nW5',code),1), :); % dried baby wipe
pen_silk = pen0(find(strcmp('W16',code),1), :); % silk

figure(1);
cmap_sweep(7, inferno);
plot(sizes, pen_n95, '.-');
hold on;
plot(sizes, pen_surg, '.-');
plot(sizes, pen_h400, '.-');
plot(sizes, pen_nmm, '.-');
plot(sizes, pen_k7, '.-');
plot(sizes, pen_dbw, '.-');
plot(sizes, pen_silk, '.-');
hold off;
set(gca, 'XScale', 'log');
%}

%}







