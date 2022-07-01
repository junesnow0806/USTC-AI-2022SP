#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <queue>
#include <string>
#include <vector>

using namespace std;
using namespace chrono;

const int MATRIX_DIM = 5;
const int MY_INFINITY = 2147483645;

// TODO:
// 获取h值的复杂度能否进一步降低？
// 在更新每个节点的path时会有vector复制开销

pair<int, int> find_vessel(const vector<vector<int>> &matrix) {
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; j < MATRIX_DIM; j++) {
            if (matrix[i][j] == 0) {
                return make_pair(i, j);
            }
        }
    }
    return make_pair(-1, -1);
}

struct node {
    vector<vector<int>> matrix;
    queue<char> path;
    char action;  // the step taken when moves from its predecessor to this node, '\0' stands for no predecessor
    int depth;
    int hvalue;
    int vessel_x;  // the row index of the vessel
    int vessel_y;  // the col index of the vessel

    node() : action('\0'), depth(0), hvalue(0), vessel_x(-1), vessel_y(-1) {
        matrix.resize(MATRIX_DIM);
        for (int i = 0; i < MATRIX_DIM; i++) {
            matrix[i].resize(MATRIX_DIM);
        }
    }
    node(const vector<vector<int>> &m, char a, int d, int hval) : matrix(m), action(a), depth(d), hvalue(hval) {
        // need to traverse the whole matrix to find the vessel
        pair<int, int> tmp = find_vessel(matrix);
        vessel_x = tmp.first;
        vessel_y = tmp.second;
    }
    node(const vector<vector<int>> &m, char a, int d, int hval, int x, int y) : matrix(m), action(a), depth(d), hvalue(hval), vessel_x(x), vessel_y(y) {}
};

struct cmp_node {
    bool operator()(const node &n1, const node &n2) {
        return (n1.depth + n1.hvalue) > (n2.depth + n2.hvalue);
    }
};

int h1(const vector<vector<int>> &start, const vector<vector<int>> &target) {
    /* heuristics 1
     * return the number of misplaced stars
     */
    int count = 0;
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; j < MATRIX_DIM; j++) {
            if (target[i][j] > 0 && start[i][j] != target[i][j])
                count++;
        }
    }
    return count;
}

int minimum(vector<int> &a) {
    int min = a[0];
    for (int i = 1; i < a.size(); i++) {
        if (a[i] < min) 
            min = a[i];
    }
    return min;
}

int h2(const vector<vector<int>> &start, const vector<vector<int>> &target) {
    /*
     * heuristic 2
     * modified Manhattan distance(MMD), considering the tunnel
     * return the summation of each node's MMD
     * 考虑到每个星球的编号唯一且绝对值为0~24的值
     * 使用星球编号绝对值作为索引, 建立一个关于各个星球在target中位置的数组
     * 可以在O(n^2)计算h2值
     */
    const int star_num = MATRIX_DIM * MATRIX_DIM;
    pair<int, int> target_pos[star_num];
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; j < MATRIX_DIM; j++) {
            int index = (target[i][j] >= 0) ? (target[i][j]) : (-target[i][j]);
            target_pos[index] = make_pair(i, j);
        }
    }
    int sum = 0;
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; j < MATRIX_DIM; j++) {
            /*
             * i: current x
             * j: current y
             * start[i][j]: the star's number
             * target_pos[abs(start[i][j])].first: the star's target x
             * target_pos[abs(start[i][j])].second: the star's target y
             * pay attention to the tunnel case!!!
             */
            vector<int> candidates;
            int star_index = abs(start[i][j]);
            int target_x = target_pos[star_index].first;
            int target_y = target_pos[star_index].second;
            int manh_dis = abs(i - target_x) + abs(j - target_y);
            candidates.push_back(manh_dis);
            /* consider the tunnels */
            // a path from (i, j) to (0, 2) then (4, 2) then to the target
            int dis = abs(i-0) + abs(j-2);
            dis += 1;
            dis += abs(4-target_x) + abs(2-target_y);
            candidates.push_back(dis);
            // a path from (i, j) to (4, 2) then (0, 2) then to the target
            dis = abs(i-4) + abs(j-2);
            dis += 1;
            dis += abs(0-target_x) + abs(2-target_y);
            candidates.push_back(dis);
            // a path from (i, j) to (2, 0) then (2, 4) then to the target
            dis = abs(i-2) + abs(j-0);
            dis += 1;
            dis += abs(2-target_x) + abs(4-target_y);
            candidates.push_back(dis);
            // a path from (i, j) to (2, 4) then (2, 0) then to the target
            dis = abs(i-2) + abs(j-4);
            dis += 1;
            dis += abs(2-target_x) + abs(0-target_y);
            candidates.push_back(dis);
            manh_dis = minimum(candidates);
            sum += manh_dis;
        }
    }
    return sum;
}

vector<node> get_successors(node &now, const vector<vector<int>> &target) {
    /* given a node and the target(used to calculate hvalue)
     * return its all successors(at most 4)
     */
    vector<node> succ;
    if (now.vessel_x > 0 && now.action != 'D' && now.matrix[now.vessel_x - 1][now.vessel_y] > 0) {
        // allowed to go up
        vector<vector<int>> new_matrix = now.matrix;
        new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x - 1][now.vessel_y];
        new_matrix[now.vessel_x - 1][now.vessel_y] = 0;
        node new_node(new_matrix, 'U', now.depth + 1, h1(new_matrix, target), now.vessel_x - 1, now.vessel_y);
        // new_node.path = now.path;
        // new_node.path.push('U');
        succ.push_back(new_node);
    } else if (now.vessel_x == 0 && now.vessel_y == 2 && now.action != 'D' && now.matrix[MATRIX_DIM - 1][now.vessel_y] > 0) {
        // tunnel, allowed to go up and turn to the last row
        vector<vector<int>> new_matrix = now.matrix;
        new_matrix[now.vessel_x][now.vessel_y] = new_matrix[MATRIX_DIM - 1][now.vessel_y];
        new_matrix[MATRIX_DIM - 1][now.vessel_y] = 0;
        node new_node(new_matrix, 'U', now.depth + 1, h1(new_matrix, target), MATRIX_DIM - 1, now.vessel_y);
        // new_node.path = now.path;
        // new_node.path.push('U');
        succ.push_back(new_node);
    }
    if (now.vessel_x < MATRIX_DIM - 1 && now.action != 'U' && now.matrix[now.vessel_x + 1][now.vessel_y] > 0) {
        // allowed to go down
        vector<vector<int>> new_matrix = now.matrix;
        new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x + 1][now.vessel_y];
        new_matrix[now.vessel_x + 1][now.vessel_y] = 0;
        node new_node(new_matrix, 'D', now.depth + 1, h1(new_matrix, target), now.vessel_x + 1, now.vessel_y);
        // new_node.path = now.path;
        // new_node.path.push('D');
        succ.push_back(new_node);
    } else if (now.vessel_x == MATRIX_DIM - 1 && now.vessel_y == 2 && now.action != 'U' && now.matrix[0][now.vessel_y] > 0) {
        // tunnel, allowed to go down and turn to the first row
        vector<vector<int>> new_matrix = now.matrix;
        new_matrix[now.vessel_x][now.vessel_y] = new_matrix[0][now.vessel_y];
        new_matrix[0][now.vessel_y] = 0;
        node new_node(new_matrix, 'D', now.depth + 1, h1(new_matrix, target), 0, now.vessel_y);
        // new_node.path = now.path;
        // new_node.path.push('D');
        succ.push_back(new_node);
    }
    if (now.vessel_y > 0 && now.action != 'R' && now.matrix[now.vessel_x][now.vessel_y - 1] > 0) {
        // allowed to go left
        vector<vector<int>> new_matrix = now.matrix;
        new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][now.vessel_y - 1];
        new_matrix[now.vessel_x][now.vessel_y - 1] = 0;
        node new_node(new_matrix, 'L', now.depth + 1, h1(new_matrix, target), now.vessel_x, now.vessel_y - 1);
        // new_node.path = now.path;
        // new_node.path.push('L');
        succ.push_back(new_node);
    } else if (now.vessel_y == 0 && now.vessel_x == 2 && now.action != 'R' && now.matrix[now.vessel_x][MATRIX_DIM - 1] > 0) {
        // tunnel, allowed to go down and turn to the last column
        vector<vector<int>> new_matrix = now.matrix;
        new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][MATRIX_DIM - 1];
        new_matrix[now.vessel_x][MATRIX_DIM - 1] = 0;
        node new_node(new_matrix, 'L', now.depth + 1, h1(new_matrix, target), now.vessel_x, MATRIX_DIM - 1);
        // new_node.path = now.path;
        // new_node.path.push('L');
        succ.push_back(new_node);
    }
    if (now.vessel_y < MATRIX_DIM - 1 && now.action != 'L' && now.matrix[now.vessel_x][now.vessel_y + 1] > 0) {
        // allowed to go right
        vector<vector<int>> new_matrix = now.matrix;
        new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][now.vessel_y + 1];
        new_matrix[now.vessel_x][now.vessel_y + 1] = 0;
        node new_node(new_matrix, 'R', now.depth + 1, h1(new_matrix, target), now.vessel_x, now.vessel_y + 1);
        // new_node.path = now.path;
        // new_node.path.push('R');
        succ.push_back(new_node);
    } else if (now.vessel_y == MATRIX_DIM - 1 && now.vessel_x == 2 && now.action != 'L' && now.matrix[now.vessel_x][0] > 0) {
        // tunnel, allowed to go down and turn to the first row
        vector<vector<int>> new_matrix = now.matrix;
        new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][0];
        new_matrix[now.vessel_x][0] = 0;
        node new_node(new_matrix, 'R', now.depth + 1, h1(new_matrix, target), now.vessel_x, 0);
        // new_node.path = now.path;
        // new_node.path.push('R');
        succ.push_back(new_node);
    }
    return succ;
}

enum htype { FIRST,
             SECOND };

int search(vector<node> &path, int g, int bound, const vector<vector<int>> &target, htype h_type) {
    /*
     * 考察path上的最后一个节点
     * 如果该节点的f值大于bound就直接返回
     * 否则递归考虑它的后继节点
     */
    node now = path.back();
    int f = 0;
    if (h_type == FIRST) {
        f = g + h1(now.matrix, target);
    } else if (h_type == SECOND) {
        f = g + h2(now.matrix, target);
    }
    if (f > bound)
        return f;
    if (now.matrix == target) {
        ofstream output;
        if (h_type == FIRST) {
            output.open("../output/output_IDA_h1.txt", ios::app);
        } else if (h_type == SECOND) {
            output.open("../output/output_IDA_h2.txt", ios::app);
        }
        for (int i = 1; i < path.size(); i++) {
            output << path[i].action;
        }
        output.close();
        return 0;
    }

    int min = MY_INFINITY;
    vector<node> successors = get_successors(now, target);
    for (auto succ : successors) {
        // if succ not in path then
        path.push_back(succ);
        int t = search(path, g + 1, bound, target, h_type);
        if (t == 0) {
            // FOUND
            return 0;
        }
        if (t < min)
            min = t;
        path.pop_back();
    }
    return min;
}

void A_h1(const vector<vector<int>> &start, const vector<vector<int>> &target) {
    /* A* algorithms using h1 as its heuristic function
     * output the action sequence in the given output file
     * need to keep all the explored nodes in memory
     */
    ofstream output("../output/output_A_h1.txt", ios::app);

    auto time_start = system_clock::now();

    priority_queue<node, vector<node>, cmp_node> fringe;
    node start_node(start, '\0', 0, h1(start, target));
    fringe.push(start_node);
    while (!fringe.empty()) {
        node now = fringe.top();
        fringe.pop();

        if (now.matrix == target) {
            auto time_end = system_clock::now();
            auto duration = duration_cast<microseconds>(time_end - time_start);
            double time_count = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            while (!now.path.empty()) {
                output << now.path.front();
                now.path.pop();
            }
            output << "," << time_count << endl;
            break;
        }

        // push children of now into fringe
        // pay attention to tunnel
        if (now.vessel_x > 0 && now.action != 'D' && now.matrix[now.vessel_x - 1][now.vessel_y] > 0) {
            // allowed to go up
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x - 1][now.vessel_y];
            new_matrix[now.vessel_x - 1][now.vessel_y] = 0;
            node new_node(new_matrix, 'U', now.depth + 1, h1(new_matrix, target), now.vessel_x - 1, now.vessel_y);
            new_node.path = now.path;
            new_node.path.push('U');
            fringe.push(new_node);
        } else if (now.vessel_x == 0 && now.vessel_y == 2 && now.action != 'D' && now.matrix[MATRIX_DIM - 1][now.vessel_y] > 0) {
            // tunnel, allowed to go up and turn to the last row
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[MATRIX_DIM - 1][now.vessel_y];
            new_matrix[MATRIX_DIM - 1][now.vessel_y] = 0;
            node new_node(new_matrix, 'U', now.depth + 1, h1(new_matrix, target), MATRIX_DIM - 1, now.vessel_y);
            new_node.path = now.path;
            new_node.path.push('U');
            fringe.push(new_node);
        }
        if (now.vessel_x < MATRIX_DIM - 1 && now.action != 'U' && now.matrix[now.vessel_x + 1][now.vessel_y] > 0) {
            // allowed to go down
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x + 1][now.vessel_y];
            new_matrix[now.vessel_x + 1][now.vessel_y] = 0;
            node new_node(new_matrix, 'D', now.depth + 1, h1(new_matrix, target), now.vessel_x + 1, now.vessel_y);
            new_node.path = now.path;
            new_node.path.push('D');
            fringe.push(new_node);
        } else if (now.vessel_x == MATRIX_DIM - 1 && now.vessel_y == 2 && now.action != 'U' && now.matrix[0][now.vessel_y] > 0) {
            // tunnel, allowed to go down and turn to the first row
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[0][now.vessel_y];
            new_matrix[0][now.vessel_y] = 0;
            node new_node(new_matrix, 'D', now.depth + 1, h1(new_matrix, target), 0, now.vessel_y);
            new_node.path = now.path;
            new_node.path.push('D');
            fringe.push(new_node);
        }
        if (now.vessel_y > 0 && now.action != 'R' && now.matrix[now.vessel_x][now.vessel_y - 1] > 0) {
            // allowed to go left
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][now.vessel_y - 1];
            new_matrix[now.vessel_x][now.vessel_y - 1] = 0;
            node new_node(new_matrix, 'L', now.depth + 1, h1(new_matrix, target), now.vessel_x, now.vessel_y - 1);
            new_node.path = now.path;
            new_node.path.push('L');
            fringe.push(new_node);
        } else if (now.vessel_y == 0 && now.vessel_x == 2 && now.action != 'R' && now.matrix[now.vessel_x][MATRIX_DIM - 1] > 0) {
            // tunnel, allowed to go down and turn to the last column
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][MATRIX_DIM - 1];
            new_matrix[now.vessel_x][MATRIX_DIM - 1] = 0;
            node new_node(new_matrix, 'L', now.depth + 1, h1(new_matrix, target), now.vessel_x, MATRIX_DIM - 1);
            new_node.path = now.path;
            new_node.path.push('L');
            fringe.push(new_node);
        }
        if (now.vessel_y < MATRIX_DIM - 1 && now.action != 'L' && now.matrix[now.vessel_x][now.vessel_y + 1] > 0) {
            // allowed to go right
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][now.vessel_y + 1];
            new_matrix[now.vessel_x][now.vessel_y + 1] = 0;
            node new_node(new_matrix, 'R', now.depth + 1, h1(new_matrix, target), now.vessel_x, now.vessel_y + 1);
            new_node.path = now.path;
            new_node.path.push('R');
            fringe.push(new_node);
        } else if (now.vessel_y == MATRIX_DIM - 1 && now.vessel_x == 2 && now.action != 'L' && now.matrix[now.vessel_x][0] > 0) {
            // tunnel, allowed to go down and turn to the first row
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][0];
            new_matrix[now.vessel_x][0] = 0;
            node new_node(new_matrix, 'R', now.depth + 1, h1(new_matrix, target), now.vessel_x, 0);
            new_node.path = now.path;
            new_node.path.push('R');
            fringe.push(new_node);
        }
    }
    // if the algorithm is right, it must find the goal, and here is nothing to do
    output.close();
}

void A_h2(const vector<vector<int>> &start, const vector<vector<int>> &target) {
    /* A* algorithms using h2 as its heuristic function
     * output the action sequence in the given output file
     * need to keep all the explored nodes in memory
     */
    ofstream output("../output/output_A_h2.txt", ios::app);

    auto time_start = system_clock::now();

    priority_queue<node, vector<node>, cmp_node> fringe;
    node start_node(start, '\0', 0, h2(start, target));
    fringe.push(start_node);
    while (!fringe.empty()) {
        node now = fringe.top();
        fringe.pop();

        if (now.matrix == target) {
            auto time_end = system_clock::now();
            auto duration = duration_cast<microseconds>(time_end - time_start);
            double time_count = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            while (!now.path.empty()) {
                output << now.path.front();
                now.path.pop();
            }
            output << "," << time_count << endl;
            break;
        }

        // push children of now into fringe
        // pay attention to tunnel
        if (now.vessel_x > 0 && now.action != 'D' && now.matrix[now.vessel_x - 1][now.vessel_y] > 0) {
            // allowed to go up
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x - 1][now.vessel_y];
            new_matrix[now.vessel_x - 1][now.vessel_y] = 0;
            node new_node(new_matrix, 'U', now.depth + 1, h2(new_matrix, target), now.vessel_x - 1, now.vessel_y);
            new_node.path = now.path;
            new_node.path.push('U');
            fringe.push(new_node);
        } else if (now.vessel_x == 0 && now.vessel_y == 2 && now.action != 'D' && now.matrix[MATRIX_DIM - 1][now.vessel_y] > 0) {
            // tunnel, allowed to go up and turn to the last row
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[MATRIX_DIM - 1][now.vessel_y];
            new_matrix[MATRIX_DIM - 1][now.vessel_y] = 0;
            node new_node(new_matrix, 'U', now.depth + 1, h2(new_matrix, target), MATRIX_DIM - 1, now.vessel_y);
            new_node.path = now.path;
            new_node.path.push('U');
            fringe.push(new_node);
        }
        if (now.vessel_x < MATRIX_DIM - 1 && now.action != 'U' && now.matrix[now.vessel_x + 1][now.vessel_y] > 0) {
            // allowed to go down
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x + 1][now.vessel_y];
            new_matrix[now.vessel_x + 1][now.vessel_y] = 0;
            node new_node(new_matrix, 'D', now.depth + 1, h2(new_matrix, target), now.vessel_x + 1, now.vessel_y);
            new_node.path = now.path;
            new_node.path.push('D');
            fringe.push(new_node);
        } else if (now.vessel_x == MATRIX_DIM - 1 && now.vessel_y == 2 && now.action != 'U' && now.matrix[0][now.vessel_y] > 0) {
            // tunnel, allowed to go down and turn to the first row
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[0][now.vessel_y];
            new_matrix[0][now.vessel_y] = 0;
            node new_node(new_matrix, 'D', now.depth + 1, h2(new_matrix, target), 0, now.vessel_y);
            new_node.path = now.path;
            new_node.path.push('D');
            fringe.push(new_node);
        }
        if (now.vessel_y > 0 && now.action != 'R' && now.matrix[now.vessel_x][now.vessel_y - 1] > 0) {
            // allowed to go left
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][now.vessel_y - 1];
            new_matrix[now.vessel_x][now.vessel_y - 1] = 0;
            node new_node(new_matrix, 'L', now.depth + 1, h2(new_matrix, target), now.vessel_x, now.vessel_y - 1);
            new_node.path = now.path;
            new_node.path.push('L');
            fringe.push(new_node);
        } else if (now.vessel_y == 0 && now.vessel_x == 2 && now.action != 'R' && now.matrix[now.vessel_x][MATRIX_DIM - 1] > 0) {
            // tunnel, allowed to go down and turn to the last column
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][MATRIX_DIM - 1];
            new_matrix[now.vessel_x][MATRIX_DIM - 1] = 0;
            node new_node(new_matrix, 'L', now.depth + 1, h2(new_matrix, target), now.vessel_x, MATRIX_DIM - 1);
            new_node.path = now.path;
            new_node.path.push('L');
            fringe.push(new_node);
        }
        if (now.vessel_y < MATRIX_DIM - 1 && now.action != 'L' && now.matrix[now.vessel_x][now.vessel_y + 1] > 0) {
            // allowed to go right
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][now.vessel_y + 1];
            new_matrix[now.vessel_x][now.vessel_y + 1] = 0;
            node new_node(new_matrix, 'R', now.depth + 1, h2(new_matrix, target), now.vessel_x, now.vessel_y + 1);
            new_node.path = now.path;
            new_node.path.push('R');
            fringe.push(new_node);
        } else if (now.vessel_y == MATRIX_DIM - 1 && now.vessel_x == 2 && now.action != 'L' && now.matrix[now.vessel_x][0] > 0) {
            // tunnel, allowed to go down and turn to the first row
            vector<vector<int>> new_matrix = now.matrix;
            new_matrix[now.vessel_x][now.vessel_y] = new_matrix[now.vessel_x][0];
            new_matrix[now.vessel_x][0] = 0;
            node new_node(new_matrix, 'R', now.depth + 1, h2(new_matrix, target), now.vessel_x, 0);
            new_node.path = now.path;
            new_node.path.push('R');
            fringe.push(new_node);
        }
    }
    // if the algorithm is right, it must find the goal, and here is nothing to do
    output.close();
}

void IDA_h1(const vector<vector<int>> &start, const vector<vector<int>> &target) {
    /* IDA* algorithms using h1 as its heuristic function
     * output the action sequence in the given output file
     * 算法使用一个栈存放当前路径
     * 先把节点入栈再判断f是否小于bound
     */
    ofstream output("../output/output_IDA_h1.txt", ios::app);
    auto time_start = system_clock::now();

    int bound = h1(start, target);
    vector<node> path;
    node start_node(start, '\0', 0, h1(start, target));
    path.push_back(start_node);
    while (true) {
        int t = search(path, 0, bound, target, htype::FIRST);
        if (t == 0) {
            // FOUND
            auto time_end = system_clock::now();
            auto duration = duration_cast<microseconds>(time_end - time_start);
            double time_count = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            output << "," << time_count << endl;
            output.close();
            return;
        }
        bound = t;
    }
}

void IDA_h2(const vector<vector<int>> &start, const vector<vector<int>> &target) {
    /* IDA* algorithms using h2 as its heuristic function
     * output the action sequence in the given output file
     * 算法使用一个栈存放当前路径
     * 先把节点入栈再判断f是否小于bound
     */
    ofstream output("../output/output_IDA_h2.txt", ios::app);
    auto time_start = system_clock::now();

    int bound = h2(start, target);
    vector<node> path;
    node start_node(start, '\0', 0, h2(start, target));
    path.push_back(start_node);
    while (true) {
        int t = search(path, 0, bound, target, htype::SECOND);
        if (t == 0) {
            // FOUND
            auto time_end = system_clock::now();
            auto duration = duration_cast<microseconds>(time_end - time_start);
            double time_count = double(duration.count()) * microseconds::period::num / microseconds::period::den;
            output << "," << time_count << endl;
            output.close();
            return;
        }
        bound = t;
    }
}

int main(int argc, char *argv[]) {
    if (argc < 4) {
        cerr << "Wrong arguments!" << endl;
        return -1;
    }

    string input_path = "../../data/" + string(argv[2]);
    ifstream input_file(input_path, ios::in);
    string target_path = "../../data/" + string(argv[3]);
    ifstream target_file(target_path, ios::in);

    /*
    string input_path = "../../data/input11.txt";
    string target_path = "../../data/target11.txt";
    ifstream input_file(input_path, ios::in);
    ifstream target_file(target_path, ios::in);
    */

    vector<vector<int>> start(MATRIX_DIM), target(MATRIX_DIM);
    for (int i = 0; i < MATRIX_DIM; i++) {
        start[i].resize(MATRIX_DIM);
        target[i].resize(MATRIX_DIM);
    }
    for (int i = 0; i < MATRIX_DIM; i++) {
        for (int j = 0; j < MATRIX_DIM; j++) {
            int temp;
            input_file >> temp;
            start[i][j] = temp;
            target_file >> temp;
            target[i][j] = temp;
        }
    }

    if (string(argv[1]) == "A_h1") {
        A_h1(start, target);
    } else if (string(argv[1]) == "A_h2") {
        A_h2(start, target);
    } else if (string(argv[1]) == "IDA_h1") {
        IDA_h1(start, target);
    } else if (string(argv[1]) == "IDA_h2") {
        IDA_h2(start, target);
    } else {
        cerr << "Wrong arguments!" << endl;
        return -1;
    }

    input_file.close();
    target_file.close();
    return 0;
}