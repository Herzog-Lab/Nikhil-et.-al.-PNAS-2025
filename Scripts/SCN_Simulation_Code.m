%% This function generates the PERIOD2 gene mRNA (M_i) using the circadian oscillator model.  
%{
Inputs : 
	Adj : Binary connectivity matrix of the network (NxN, N: network size)
	sig : Coupling matrix (NxN, N: network size). The (i,j)th element is \alpha_{ij} (Method section)
	Days : Number of days to simulate the network model 
	Samping_rate : Rate of data sampling (in hours)

Outputs : PERIOD2 gene mRNA (M_i) for all the network nodes (an array of size : (Sampling_rate*Days*24)+1 x N)

%}

function Mc = Generate_simulated_data(Adj, sig, Days, Sampling_rate)


	t = 0:Sampling_rate:24*Days;                % Time vector in hours 


	%% Parameters
	N = size(Adj,1);                 % Number of circadian oscillator
	param.sig = sig;                 % Coupling strength
	param.N  = N;                  % Number of oscillators
	param.Adj = Adj;                 % Adjacency matrix
	

	%% Initial conditions
	% Random initial states for 3 variables per oscillator
	a = 0.8; b = 4.2;
	a2 = 5; b2 = 1;
	a3 = 6; b3 = 2;

	r1 = a + (b-a).*rand(N,1);
	r2 = (a2-b2).*rand(N,1);
	r3 = (a3-b3).*rand(N,1);

	x0 = [r1; r2; r3];

	%% Model constants
	param.vm = linspace(0.345,0.395,N);  
	param.vs0 = 0.73;
	param.K1 = 1.0; param.Km = 0.5; param.ks = 0.417; 
	param.vd = 1.167; param.Kd = 0.13; param.k1 = 0.417; 
	param.k2 = 0.5; param.n = 4;



	% Simulate the network
	options = odeset('refine', 2, 'RelTol', 1e-10, 'AbsTol', 1e-12);
	[t, x] = ode45(@(t,X) Circ_Oscillator_network(t,X,param), t, x0, options);

	% Extract variables
	Mc = x(:,1:N); 
	Pc = x(:,N+1:2*N); 
	Pn = x(:,2*N+1:end);

end


% Ode function -----------------------------
function dx = Circ_Oscillator_network(t, x, param)
	N = param.N;
	Adj = param.Adj;
	sig = param.sig;

	% State variables
	M  = x(1:N)';
	Pc = x(N+1:2*N)';
	Pn = x(2*N+1:3*N)';

	% Coupling interactions
	[M_i, M_j] = meshgrid(M);
	Dx = M_j - M_i;
	InterActionFunction = sum(sig .* Adj .* Dx, 1);

	% Control variables
	vm = param.vm;
	vs = param.vs0' + InterActionFunction;

	% Model equations
	dM  = -vm .* M ./ (param.Km + M) + param.K1^param.n ./ (param.K1^param.n + Pn.^param.n) .* vs;
	dPc = param.ks .* M - param.vd .* Pc ./ (param.Kd + Pc) - param.k1 .* Pc + param.k2 .* Pn;
	dPn = param.k1 .* Pc - param.k2 .* Pn;

	dx = [dM'; dPc'; dPn'];
end




