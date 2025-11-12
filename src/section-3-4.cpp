#include <algorithm>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace std;

const int PASS{0};
const int BET{1};
const int NUM_ACTIONS{2};

class Node {
   public:
    string infoset;
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
        for (int a = 0; a < NUM_ACTIONS; a++) avg[a] = (sum > 0) ? strategy_sum[a] / sum : 1.0 / NUM_ACTIONS;
        return avg;
    }

    string describe() {
        string s = infoset + ": [";
        vector<double> avg = get_average_strategy(strategy_sum);

        for (size_t i = 0; i < avg.size(); i++) {
            s += to_string(avg[i]);
            if (i + 1 < avg.size()) s += ", ";
        }
        s += "]";

        return s;
    }
};

class KuhnPoker {
   private:
    mt19937 generator{0};
    unordered_map<string, unique_ptr<Node>> node_map = unordered_map<string, unique_ptr<Node>>();

    void shuffle(vector<int>& cards) {
        for (int i = cards.size() - 1; i > 0; --i) {
            uniform_int_distribution<> dist(0, i);
            int j = dist(generator);
            swap(cards[i], cards[j]);
        }
    }

    double cfr(vector<int> cards, string history, double p0, double p1) {
        int plays = history.length();
        int player = plays % 2;
        int opponent = 1 - player;

        if (plays > 1) {
            bool terminalPass = history[plays - 1] == 'p';
            bool doubleBet = history.substr(plays - 2, 2) == "bb";
            bool isPlayerCardHigher = cards[player] > cards[opponent];

            if (terminalPass) {
                if (history == "pp")
                    return isPlayerCardHigher ? 1.0 : -1.0;
                else
                    return 1.0;
            } else if (doubleBet) {
                return isPlayerCardHigher ? 2.0 : -2.0;
            }
        }

        string infoset = to_string(cards[player]) + history;

        auto& node = node_map.try_emplace(infoset, make_unique<Node>()).first->second;
        node->infoset = infoset;

        vector<double> strategy = node->get_strategy(player == 0 ? p0 : p1);
        vector<double> util = vector<double>(NUM_ACTIONS);
        double nodeUtil = 0;

        for (int a = 0; a < NUM_ACTIONS; a++) {
            string nextHistory = history + (a == 0 ? "p" : "b");
            util[a] = player == 0 ? -cfr(cards, nextHistory, p0 * strategy[a], p1)
                                  : -cfr(cards, nextHistory, p0, p1 * strategy[a]);
            nodeUtil += strategy[a] * util[a];
        }

        for (int a = 0; a < NUM_ACTIONS; a++) {
            double regret = util[a] - nodeUtil;
            node->regret_sum[a] += (player == 0 ? p1 : p0) * regret;
        }

        return nodeUtil;
    }

   public:
    void train(int iterations) {
        vector<int> cards{1, 2, 3};
        double util = 0;
        for (int i = 0; i < iterations; i++) {
            shuffle(cards);
            util += cfr(cards, "", 1.0, 1.0);
        }

        cout << "Average game value: " << util / iterations << endl;
        for (const auto& [key, node] : node_map) {
            cout << node->describe() << endl;
        }
    }
};

int main() {
    KuhnPoker solver = KuhnPoker();
    solver.train(1'000'000);

    return 0;
}
