// Section 3.5 (2)
// Two-player Counterfactual Regret Minimization (CFR) with chance sampling for the last round of Dudo.
// Implemented with full history vector passing.
// Exercise from "An Introduction to Counterfactual Regret Minimization" by Todd W. Neller and Marc Lanctot

#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

const int NUM_SIDES = 6, NUM_ACTIONS = (2 * NUM_SIDES) + 1, DUDO = NUM_ACTIONS - 1;
const int CLAIM_NUM[] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2};
const int CLAIM_RANK[] = {2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 1};

class Node {
   public:
    int id;
    vector<double> regret_sum = vector<double>(NUM_ACTIONS, 0.0), strategy = vector<double>(NUM_ACTIONS, 0.0),
                   strategy_sum = vector<double>(NUM_ACTIONS, 0.0);

    vector<double> get_strategy(double realization_weight) {
        double sum = 0.0;

        for (int a = 0; a < NUM_ACTIONS; a++) {
            strategy[a] = regret_sum[a] > 0 ? regret_sum[a] : 0;
            sum += strategy[a];
        }

        for (int a = 0; a < NUM_ACTIONS; a++) {
            if (sum > 0)
                strategy[a] /= sum;
            else
                strategy[a] = 1.0 / NUM_ACTIONS;

            strategy_sum[a] += realization_weight * strategy[a];
        }

        return strategy;
    }

    vector<double> get_average_strategy(const vector<double>& strategy_sum) {
        vector<double> avg(NUM_ACTIONS, 0.0);
        double sum = accumulate(strategy_sum.begin(), strategy_sum.end(), 0.0);

        for (int a = 0; a < NUM_ACTIONS; a++) {
            avg[a] = (sum > 0) ? strategy_sum[a] / sum : 1.0 / NUM_ACTIONS;
        }

        return avg;
    }
};

class DudoTrainer {
   private:
    mt19937 generator{0};
    unordered_map<int, unique_ptr<Node>> node_map = unordered_map<int, unique_ptr<Node>>();

    int get_infoset_key(int player_roll, const vector<int>& history) {
        int infoset_num = player_roll;

        for (int a = NUM_ACTIONS - 2; a >= 0; a--) {
            bool is_claimed = false;
            for (int action : history) {
                if (action == a) {
                    is_claimed = true;
                    break;
                }
            }

            infoset_num = 2 * infoset_num + (is_claimed ? 1 : 0);
        }

        return infoset_num;
    }

    int count_matches(const vector<int>& dice, int rank) {
        int count = 0;

        for (int d : dice) {
            if (d == rank || d == 1) count++;
        }

        return count;
    }

    void roll(vector<int>& dice) {
        uniform_int_distribution<> dist(1, 6);

        for (int& die : dice) {
            die = dist(generator);
        }
    }

    double cfr(vector<int> dice, vector<int>& history, double p0, double p1) {
        int turn = history.size();
        int player = turn % 2;

        if (!history.empty() && history.back() == DUDO) {
            int challenged_action = history[history.size() - 2];

            int claim_num = CLAIM_NUM[challenged_action];
            int claim_rank = CLAIM_RANK[challenged_action];
            int count = count_matches(dice, claim_rank);

            bool claimant_wins = (count >= claim_num);
            int claimant = (turn - 2) % 2;

            if (claimant_wins)
                return (claimant == 0) ? 1.0 : -1.0;
            else
                return (claimant == 0) ? -1.0 : 1.0;
        }

        int key = get_infoset_key(dice[player], history);
        auto& node = node_map.try_emplace(key, make_unique<Node>()).first->second;
        node->id = key;

        vector<double> strategy = node->get_strategy(player == 0 ? p0 : p1);
        vector<double> utility = vector<double>(NUM_ACTIONS);
        double node_utility = 0;

        int last_action = history.empty() ? -1 : history.back();

        for (int a = 0; a < NUM_ACTIONS; a++) {
            if (a == DUDO && history.empty()) continue;
            if (a != DUDO && !history.empty() && a <= last_action) continue;

            history.push_back(a);
            utility[a] = cfr(dice, history, player == 0 ? p0 * strategy[a] : p0, player == 1 ? p1 * strategy[a] : p1);
            history.pop_back();

            node_utility += strategy[a] * utility[a];
        }

        for (int a = 0; a < NUM_ACTIONS; a++) {
            if (a == DUDO && history.empty()) continue;
            if (a != DUDO && !history.empty() && a <= last_action) continue;

            double regret;
            if (player == 0) {
                regret = utility[a] - node_utility;
                node->regret_sum[a] += p1 * regret;
            } else {
                regret = node_utility - utility[a];
                node->regret_sum[a] += p0 * regret;
            }
        }

        return node_utility;
    }

   public:
    void train(int iterations) {
        vector<int> dice{0, 0};
        vector<int> history;
        double total_utility = 0;

        for (int i = 0; i < iterations; i++) {
            roll(dice);
            total_utility += cfr(dice, history, 1.0, 1.0);
        }

        cout << "Average game value: " << total_utility / iterations << endl;
    }
};

int main() {
    DudoTrainer solver = DudoTrainer();
    solver.train(100'000);

    return 0;
}
