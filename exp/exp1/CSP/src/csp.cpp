#include <fstream>
#include <iostream>
#include <string>
#include <vector>

using namespace std;

const int LEAST_ONDUTY_WORKERS_NUM = 5;     // 每天要求的最少值班人数
const int LEAST_RESTDAY_NUM = 2;            // 工人一周最少的休息天数
const int MOST_CONTINUOUS_RESTDAY_NUM = 2;  // 工人一周连续休息的最多天数
const int LEAST_SENIOR_WORKERS_NUM = 1;     // 每天要求值班的最少senior人数

enum grade {
    junior,
    senior
};  // 工人级别

enum st {
    none,
    duty,
    rest
};  // 变量的取值: 值班, 休息, 未赋值

enum wst {
    both,
    d,
    r
};  // worker status, 表示一个工人在某一天是必须值班, 还是必须休息, 还是两种都可以

bool check_cons1(vector<vector<st>> &state) {
    /* 检查是否满足约束1: 每个工人每周必须休息2天或以上 */
    for (int i = 1; i < state.size(); i++) {
        int count = 0;
        for (int j = 0; j < 7; j++) {
            if (state[i][j] == st::rest)
                count++;
        }
        if (count < LEAST_RESTDAY_NUM)
            return false;
    }
    return true;
}

bool check_cons2(vector<vector<st>> &state) {
    /* 检查是否满足约束2: 每个工人不可以连续休息3天(不考虑跨周情况)
     * 遍历state矩阵来判断
     */
    for (int i = 1; i < state.size(); i++) {
        for (int j = 0; j < 7; j++) {
            if (state[i][j] == st::rest) {
                int count = 1;
                j++;
                while (j < 7 && state[i][j] == st::rest) {
                    count++;
                    j++;
                }
                if (count > MOST_CONTINUOUS_RESTDAY_NUM)
                    return false;
            }
        }
    }
    return true;
}

bool check_cons3_4(vector<vector<st>> &state) {
    /* 检查是否满足约束3: 每天都有指定数量的工人值班
     */
    for (int j = 0; j < 7; j++) {
        int count = 0;
        for (int i = 1; i < state.size(); i++) {
            if (state[i][j] == st::duty)
                count++;
        }
        if (count < LEAST_ONDUTY_WORKERS_NUM)
            return false;
    }
    return true;
}

bool check_cons5(vector<vector<st>> &state, vector<int> &seniors) {
    /* 检查是否满足约束5: 每天至少要指定数量的senior工人值班
     */
    for (int i = 0; i < 7; i++) {
        int count = 0;
        for (auto sno : seniors) {
            if (state[sno][i] == st::duty)
                count++;
        }
        if (count < LEAST_SENIOR_WORKERS_NUM)
            return false;
    }
    return true;
}

bool check_conflict(vector<vector<st>> &state, vector<pair<int, int>> &conflict_pairs) {
    /* 检查是否满足工人矛盾方面的约束 */
    for (auto &conpair : conflict_pairs) {
        int w1 = conpair.first;
        int w2 = conpair.second;
        for (int j = 0; j < 7; j++) {
            if (state[w1][j] == st::duty && state[w2][j] == st::duty)
                return false;
        }
    }
    return true;
}

bool goal_test(vector<vector<st>> &state, vector<int> &seniors, vector<pair<int, int>> &conflict_pairs) {
    /* 判断给定状态是否以达成目标 */
    if (!check_cons1(state)) {
        return false;
    }
    if (!check_cons2(state)) {
        return false;
    }
    if (!check_cons3_4(state)) {
        return false;
    }
    if (!check_cons5(state, seniors)) {
        return false;
    }
    if (!check_conflict(state, conflict_pairs)) {
        return false;
    }
    return true;
}

int count_workday(vector<vector<st>> &state, int wno) {
    /* 计算当前赋值状态下工人wno的值班天数 */
    int count = 0;
    for (int i = 0; i < 7; i++) {
        if (state[wno][i] == st::duty)
            count++;
    }
    return count;
}

int count_continuous_restday_num(vector<vector<st>> &state, int wno) {
    /* 计算工人wno连续休息的天数 */
    int max_count = 0;
    for (int i = 0; i < 7; i++) {
        if (state[wno][i] == st::rest) {
            int count = 1;
            i++;
            while (i < 7 && state[wno][i] == st::rest) {
                count++;
                i++;
            }
            if (count > max_count)
                max_count = count;
        }
    }
    return max_count;
}

int count_workers_day(vector<vector<st>> &state, int day) {
    /* 计算一天已赋值工人中值班的人数 */
    int count = 0;
    for (int i = 1; i < state.size() && state[i][day] != st::none; i++) {
        if (state[i][day] == st::duty) {
            count++;
        }
    }
    return count;
}

int count_senior_day(vector<vector<st>> &state, int wno, int day, vector<grade> &grades) {
    /* 计算一天中已赋值中值班senior的人数 */
    int count = 0;
    for (int i = 1; i <= wno; i++) {
        if (grades[i] == grade::senior && state[i][day] == st::duty)
            count++;
    }
    return count;
}

bool update_workers_status(vector<vector<st>> &state, vector<vector<wst>> &workers_status, int wno, int day,
                           vector<grade> &grades, vector<int> &seniors, vector<pair<int, int>> &conflict_pairs) {
    /*
     * state[wno][day]是刚被赋值的元素
     * 更新工人可用状态表
     * 要考虑的方面:
     * 工人是否已经值班了5天, 要保重休息天数
     * 工人是否已经连续休息了2天, 要预防连续三天休息
     * 看当天是否能保证可以找到至少5个工人值班
     * 看当天是不是已经有senior工人值班了
     * 该工人在这天值班了, 那与他有矛盾的工人就不能在这天值班了
     * 当修改一个工人在某一天的状态为wst::d时, 与他有矛盾的工人在这天的状态都要置为wst::r
     * 当修改一个工人在某一天的状态为wst::d(r)时, 如果发现他的状态已经是r(d)了, 说明有冲突, 直接返回更新失败
     */
    int worker_num = state.size() - 1;
    if (state[wno][day] == st::duty) {
        /* 该工人被安排在day值班后更新workers_status */

        // 1
        int workday_num = count_workday(state, wno);
        if (workday_num == 7 - LEAST_RESTDAY_NUM) {
            // 工作天数已达上限, 后面不能让他值班了
            for (int i = day + 1; i < 7; i++) {
                if (workers_status[wno][i] != wst::d)
                    workers_status[wno][i] = wst::r;
                else
                    return false;
            }
        } else if (workday_num > 7 - LEAST_RESTDAY_NUM) {
            cerr << "压榨工人了!!!" << endl;
        }
        // 3
        int non_assign_worker_num = worker_num - wno;        // 当天还没有赋值的工人变量数
        int worker_num_day = count_workers_day(state, day);  // 当天已赋值工人中的值班人数
        if (worker_num_day + non_assign_worker_num == LEAST_ONDUTY_WORKERS_NUM) {
            // 已赋值且赋值为值班的人数+未赋值的人数 == 最少值班工人数
            // 剩下的未赋值工人必须都值班
            // 剩下的未赋值工人中有可能有矛盾
            for (auto cp : conflict_pairs) {
                if (state[cp.first][day] == st::none && state[cp.second][day] == st::none) {
                    return false;
                }
            }
            for (int i = wno + 1; i < state.size(); i++) {
                if (workers_status[i][day] != wst::r)
                    workers_status[i][day] = wst::d;
                else
                    return false;
                for (auto cp : conflict_pairs) {
                    if (i == cp.first) {
                        if (workers_status[cp.second][day] != wst::d)
                            workers_status[cp.second][day] = wst::r;
                        else
                            return false;
                    } else if (i == cp.second) {
                        if (workers_status[cp.first][day] != wst::d)
                            workers_status[cp.first][day] = wst::r;
                        else
                            return false;
                    }
                }
            }
        }
        // 4
        // 如果除了一个senior工人外的其他senior工人都已经赋值为false或wst为rest
        // 那该senior工人在这天就一定要值班
        int assigned_duty_num = count_senior_day(state, wno, day, grades);
        if (assigned_duty_num < LEAST_SENIOR_WORKERS_NUM) {
            int count = 0;
            for (auto sno : seniors) {
                if (sno <= wno || workers_status[sno][day] == wst::r)
                    count++;
            }
            if (seniors.size() - count == LEAST_SENIOR_WORKERS_NUM - assigned_duty_num) {
                for (auto sno : seniors) {
                    if (sno > wno && workers_status[sno][day] != wst::r) {
                        workers_status[sno][day] = wst::d;
                        for (auto cp : conflict_pairs) {
                            if (sno == cp.first) {
                                if (workers_status[cp.second][day] != wst::d)
                                    workers_status[cp.second][day] = wst::r;
                                else
                                    return false;
                            } else if (sno == cp.second) {
                                if (workers_status[cp.first][day] != wst::d)
                                    workers_status[cp.first][day] = wst::r;
                                else
                                    return false;
                            }
                        }
                    }
                }
            }
        }
        // 5
        for (auto cp : conflict_pairs) {
            if (wno == cp.first) {
                int another = cp.second;
                if (workers_status[another][day] != wst::d)
                    workers_status[another][day] = wst::r;
                else
                    return false;
            } else if (wno == cp.second) {
                int another = cp.first;
                if (workers_status[another][day] != wst::d)
                    workers_status[another][day] = wst::r;
                else
                    return false;
            }
        }
    } else if (state[wno][day] == st::rest) {
        // 2
        int cont_rday_num = count_continuous_restday_num(state, wno);
        if (cont_rday_num == MOST_CONTINUOUS_RESTDAY_NUM) {
            // 已经连续休息2天了, 下一天不能休息了
            if (workers_status[wno][day + 1] != wst::r)
                workers_status[wno][day + 1] = wst::d;
            else
                return false;
            for (auto cp : conflict_pairs) {
                if (wno == cp.first) {
                    if (workers_status[cp.second][day + 1] != wst::d)
                        workers_status[cp.second][day + 1] = wst::r;
                    else
                        return false;
                } else if (wno == cp.second) {
                    if (workers_status[cp.first][day + 1] != wst::d)
                        workers_status[cp.first][day + 1] = wst::r;
                    else
                        return false;
                }
            }
        } else if (cont_rday_num > MOST_CONTINUOUS_RESTDAY_NUM) {
            cerr << "连续休息太多了!" << endl;
        }
        // 3
        int non_assign_worker_num = worker_num - wno;
        int worker_num_day = count_workers_day(state, day);
        if (worker_num_day + non_assign_worker_num == LEAST_ONDUTY_WORKERS_NUM) {
            // 已赋值且赋值为值班的人数+未赋值的人数 == 最少值班工人数
            // 剩下的未赋值工人必须都值班
            // 剩下的未赋值工人中有可能有矛盾
            for (auto cp : conflict_pairs) {
                if (state[cp.first][day] == st::none && state[cp.second][day] == st::none) {
                    return false;
                }
            }
            for (int i = wno + 1; i < state.size(); i++) {
                if (workers_status[i][day] != wst::r)
                    workers_status[i][day] = wst::d;
                else
                    return false;
                for (auto cp : conflict_pairs) {
                    if (i == cp.first) {
                        if (workers_status[cp.second][day] != wst::d)
                            workers_status[cp.second][day] = wst::r;
                        else
                            return false;
                    } else if (i == cp.second) {
                        if (workers_status[cp.first][day] != wst::d)
                            workers_status[cp.first][day] = wst::r;
                        else
                            return false;
                    }
                }
            }
        }
        // 4
        // 如果除了一个senior工人外的其他senior工人都已经赋值为false或wst为rest
        // 那该senior工人在这天就一定要值班
        int assigned_duty_num = count_senior_day(state, wno, day, grades);
        if (assigned_duty_num < LEAST_SENIOR_WORKERS_NUM) {
            int count = 0;
            for (auto sno : seniors) {
                if (sno <= wno || workers_status[sno][day] == wst::r)
                    count++;
            }
            if (seniors.size() - count == LEAST_SENIOR_WORKERS_NUM - assigned_duty_num) {
                for (auto sno : seniors) {
                    if (sno > wno && workers_status[sno][day] != wst::r) {
                        workers_status[sno][day] = wst::d;
                        for (auto cp : conflict_pairs) {
                            if (sno == cp.first) {
                                if (workers_status[cp.second][day] != wst::d)
                                    workers_status[cp.second][day] = wst::r;
                                else
                                    return false;
                            } else if (sno == cp.second) {
                                if (workers_status[cp.first][day] != wst::d)
                                    workers_status[cp.first][day] = wst::r;
                                else
                                    return false;
                            }
                        }
                    }
                }
            }
        }
    }
    return true;
}

bool assign(vector<vector<st>> &state, vector<vector<wst>> &workers_status, int wno, int day,
            vector<grade> &grades, vector<int> &seniors, vector<pair<int, int>> &conflict_pairs) {
    /* 考察某一天day对某个工人wno的指派
     * 先赋值再判断赋值后是否合法
     * 如果是根据维护的workers_status来赋值
     * 那可以做的赋值都是合法的才对
     * 当且仅当day = 6且wno等于最后一个工人时表示已经全部赋值完了
     * 然后再用goal_test
     */
    int worker_num = state.size() - 1;
    if (workers_status[wno][day] == wst::both) {
        /* 该工人在这天可以值班也可以休息
         * 保存状态, 先深度优先搜索值班的情况
         * 返回后恢复状态, 深度优先搜索休息的情况
         */
        vector<vector<wst>> wst_bak = workers_status;
        vector<vector<st>> state_bak = state;
        state[wno][day] = st::duty;
        if (wno == worker_num && day == 6) {
            if (goal_test(state, seniors, conflict_pairs))
                return true;
        }
        // 更新workers_status
        bool ret = update_workers_status(state, workers_status, wno, day, grades, seniors, conflict_pairs);
        if (ret == false)
            return false;
        // 如果是合法赋值就进行下一个工人或工作日的赋值
        int next_wno = wno + 1;
        int next_day = day;
        if (wno == worker_num) {
            next_wno = 1;
            next_day = (day + 1) % 7;
        }
        bool result = assign(state, workers_status, next_wno, next_day, grades, seniors, conflict_pairs);
        if (result == true)
            return true;
        workers_status = wst_bak;
        state = state_bak;

        state[wno][day] = st::rest;
        if (wno == worker_num && day == 6) {
            if (goal_test(state, seniors, conflict_pairs))
                return true;
        }
        // 更新workers_status
        ret = update_workers_status(state, workers_status, wno, day, grades, seniors, conflict_pairs);
        if (ret == false)
            return false;
        // 如果是合法赋值就进行下一个工人或工作日的赋值
        next_wno = wno + 1;
        next_day = day;
        if (wno == worker_num) {
            next_wno = 1;
            next_day = (day + 1) % 7;
        }
        result = assign(state, workers_status, next_wno, next_day, grades, seniors, conflict_pairs);
        if (result == true)
            return true;
        workers_status = wst_bak;
        state = state_bak;
    } else if (workers_status[wno][day] == wst::d) {
        // 该工人在这天必须值班
        vector<vector<wst>> wst_bak = workers_status;
        vector<vector<st>> state_bak = state;
        state[wno][day] = st::duty;
        if (wno == worker_num && day == 6) {
            if (goal_test(state, seniors, conflict_pairs))
                return true;
        }
        // 更新workers_status
        bool ret = update_workers_status(state, workers_status, wno, day, grades, seniors, conflict_pairs);
        if (!ret)
            return false;
        // 如果是合法赋值就进行下一个工人或工作日的赋值
        int next_wno = wno + 1;
        int next_day = day;
        if (wno == worker_num) {
            next_wno = 1;
            next_day = (day + 1) % 7;
        }
        bool result = assign(state, workers_status, next_wno, next_day, grades, seniors, conflict_pairs);
        if (result)
            return true;
        workers_status = wst_bak;
        state = state_bak;
    } else if (workers_status[wno][day] == wst::r) {
        // 该工人在这天必须休息
        vector<vector<wst>> wst_bak = workers_status;
        vector<vector<st>> state_bak = state;
        state[wno][day] = st::rest;
        if (wno == worker_num && day == 6) {
            if (goal_test(state, seniors, conflict_pairs))
                return true;
        }
        // 更新workers_status
        bool ret = update_workers_status(state, workers_status, wno, day, grades, seniors, conflict_pairs);
        if (!ret)
            return false;
        // 如果是合法赋值就进行下一个工人或工作日的赋值
        int next_wno = wno + 1;
        int next_day = day;
        if (wno == worker_num) {
            next_wno = 1;
            next_day = (day + 1) % 7;
        }
        bool result = assign(state, workers_status, next_wno, next_day, grades, seniors, conflict_pairs);
        if (result)
            return true;
        workers_status = wst_bak;
        state = state_bak;
    }
    return false;
}

int main() {
    ifstream input("../input/input2.txt", ios::in);
    int worker_num = 0;
    input >> worker_num;
    vector<grade> grades(worker_num + 1);
    vector<int> seniors;  // 记录senior工人的编号
    vector<int> juniors;
    for (int i = 1; i <= worker_num; i++) {
        string tmp;
        input >> tmp;
        if (tmp == "senior") {
            grades[i] = grade::senior;
            seniors.push_back(i);
        } else if (tmp == "junior") {
            grades[i] = grade::junior;
            juniors.push_back(i);
        } else
            cerr << "Wrong worker grade input: " << i << endl;
    }
    int conflict_pairs_num;
    input >> conflict_pairs_num;
    vector<pair<int, int>> conflict_pairs(conflict_pairs_num);
    for (int i = 0; i < conflict_pairs.size(); i++) {
        int worker1, worker2;
        input >> worker1 >> worker2;
        conflict_pairs[i] = make_pair(worker1, worker2);
    }

    vector<vector<st>> state(worker_num + 1);
    for (int i = 0; i < worker_num + 1; i++) {
        state[i].resize(7);
    }
    vector<vector<wst>> workers_status(worker_num + 1);  // 标记每个工人在某一天是否可用
    for (int i = 0; i < worker_num + 1; i++) {
        workers_status[i].resize(7);
    }

    bool result = assign(state, workers_status, 1, 0, grades, seniors, conflict_pairs);
    if (result == true) {
        ofstream output("../output/output2.txt", ios::out);
        /*
        for (int i = 1; i < state.size(); i++) {
            for (int j = 0; j < 7; j++) {
                if (state[i][j] == st::duty) {
                    output << 1 << " ";
                } else if (state[i][j] == st::rest) {
                    output << 0 << " ";
                } else {
                    cerr << "???" << endl;
                }
            }
            output << endl;
        }
        */
        for (int j = 1; j <= 6; j++) {
            for (int i = 1; i < state.size(); i++) {
                if (state[i][j] == st::duty) {
                    output << i << " ";
                }
            }
            output << endl;
        }
        for (int i = 1; i < state.size(); i++) {
            if (state[i][0] == st::duty) {
                output << i << " ";
            }
        }
        output.close();
    } else {
        cout << "Could not find a solution." << endl;
    }

    input.close();
    return 0;
}