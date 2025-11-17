#include <algorithm>
#include <fstream>
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

    string claim_history_to_string(const vector<bool>& is_claimed) {
        string str = "";
        for (int a = 0; a < NUM_ACTIONS; a++)
            if (is_claimed[a]) {
                if (str.length() > 0) str.append(",");

                str.append(to_string(CLAIM_NUM[a]));
                str.append("*");
                str.append(to_string(CLAIM_RANK[a]));
            }

        return str;
    }

    int get_infoset_key(int player_roll, const vector<bool>& is_claimed) {
        int infoset_num = player_roll;

        for (int a = NUM_ACTIONS - 2; a >= 0; a--) {
            infoset_num = 2 * infoset_num + (is_claimed[a] ? 1 : 0);
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

    double cfr(vector<int> dice, vector<bool>& is_claimed, int last_action, int turn, double p0, double p1) {
        int player = turn % 2;

        if (is_claimed[DUDO]) {
            int challenged_claim = -1;
            for (int a = DUDO - 1; a >= 0; a--) {
                if (is_claimed[a]) {
                    challenged_claim = a;
                    break;
                }
            }

            int claim_num = CLAIM_NUM[challenged_claim];
            int claim_rank = CLAIM_RANK[challenged_claim];
            int count = count_matches(dice, claim_rank);
            bool claimant_wins = (count >= claim_num);

            int claimant = turn % 2;

            if (claimant_wins)
                return (claimant == 0 ? 1.0 : -1.0);
            else
                return (claimant == 0 ? -1.0 : 1.0);
        }

        int key = get_infoset_key(dice[player], is_claimed);
        auto& node = node_map.try_emplace(key, make_unique<Node>()).first->second;
        node->id = key;

        vector<double> strategy = node->get_strategy(player == 0 ? p0 : p1);
        vector<double> utility = vector<double>(NUM_ACTIONS);
        double node_utility = 0;

        for (int a = 0; a < NUM_ACTIONS; a++) {
            if (a == DUDO) {
                if (turn == 0) continue;
            } else {
                if (a <= last_action && turn > 0) continue;
            }

            is_claimed[a] = true;

            utility[a] = cfr(dice, is_claimed, a, turn + 1, player == 0 ? p0 * strategy[a] : p0,
                             player == 1 ? p1 * strategy[a] : p1);

            is_claimed[a] = false;

            node_utility += strategy[a] * utility[a];
        }

        for (int a = 0; a < NUM_ACTIONS; a++) {
            if (a == DUDO && turn == 0) continue;
            if (a != DUDO && a <= last_action && turn > 0) continue;

            double regret = (player == 0) ? (utility[a] - node_utility) : (node_utility - utility[a]);
            node->regret_sum[a] += (player == 0 ? p1 : p0) * regret;
        }

        return node_utility;
    }

   public:
    void train(int iterations) {
        vector<int> dice{0, 0};
        vector<bool> is_claimed(NUM_ACTIONS, false);
        double total_utility = 0;

        for (int i = 0; i < iterations; i++) {
            roll(dice);
            fill(is_claimed.begin(), is_claimed.end(), false);
            total_utility += cfr(dice, is_claimed, -1, 0, 1.0, 1.0);
        }

        cout << "Average game value: " << total_utility / iterations << endl;
    }

    void save_strategies(const string& filename) {
        ofstream outfile(filename);

        if (!outfile.is_open()) {
            cerr << "Error: Could not open file " << filename << " for writing." << endl;
            return;
        }

        vector<int> keys;
        keys.reserve(node_map.size());
        for (auto& pair : node_map) {
            keys.push_back(pair.first);
        }
        sort(keys.begin(), keys.end());

        for (int key : keys) {
            Node* node = node_map[key].get();
            vector<double> avg_strategy = node->get_average_strategy(node->strategy_sum);

            int shift = NUM_ACTIONS - 1;
            int roll = key >> shift;

            vector<bool> is_claimed(NUM_ACTIONS, false);
            for (int a = 0; a < shift; a++) {
                if ((key >> a) & 1) {
                    is_claimed[a] = true;
                }
            }

            string history = claim_history_to_string(is_claimed);
            if (history.empty()) history = "(Start)";

            outfile << "Roll: " << roll << " | History: " << history << "\n";
            outfile << "    Strategy: ";

            bool first = true;
            for (int a = 0; a < NUM_ACTIONS; a++) {
                if (avg_strategy[a] > 0.001) {
                    if (!first) outfile << ", ";

                    if (a == DUDO) {
                        outfile << "Dudo";
                    } else {
                        outfile << CLAIM_NUM[a] << "x" << CLAIM_RANK[a];
                    }

                    outfile << ": " << fixed << setprecision(2) << avg_strategy[a] * 100 << '%';
                    first = false;
                }
            }
            outfile << "\n" << endl;
        }

        outfile.close();
    }
};

int main() {
    DudoTrainer solver = DudoTrainer();
    solver.train(1'000'000);
    solver.save_strategies("strategies.txt");

    return 0;
}
