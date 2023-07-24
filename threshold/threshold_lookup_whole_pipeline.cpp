#include "realtime_heap_allocate.hpp"
//#include "pisa/tools/app.hpp"
#include <random>
#include <sstream>
#include <cmath>
#include <iostream>
#include <optional>
#include <unordered_set>
#include <queue>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <sparsehash/dense_hash_set>
#include <sparsehash/dense_hash_map>
#include <bits/stdc++.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <numeric>

#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/split.hpp"
#include <boost/functional/hash.hpp>
#include <boost/interprocess/file_mapping.hpp>
#include <boost/interprocess/mapped_region.hpp>

#include "mio/mmap.hpp"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/spdlog.h"

#include "mappable/mapper.hpp"

#include "app.hpp"
#include "cursor/max_scored_cursor.hpp"
#include "index_types.hpp"
#include "io.hpp"
#include "query/algorithm.hpp"
#include "util/util.hpp"
#include "wand_data_compressed.hpp"
#include "wand_data_raw.hpp"

#include "query/algorithm.hpp"
#include "query/queries.hpp"
#include "scorer/scorer.hpp"

#include "CLI/CLI.hpp"

#include "block_freq_index.hpp"
#include "codec/simdbp.hpp"
#include "io.hpp"
#include "timer.hpp"
using namespace pisa;
using namespace std;

#define SHORTQ 10000
#define LONGQ 20000

#define SHORT_L 10000
#define SHORT_U 20000

#define MIDDLE_L 100000
#define MIDDLE_U 200000

#define LONG_L 1000000
#define LONG_U 2000000

#define NUM_LOOKUP 50

#define MaxQueryLen 16

using namespace pisa;
namespace bip = boost::interprocess;
using term_id_type = uint32_t;
//using ext::hash;
using google::dense_hash_set;
using google::dense_hash_map;


using wand_raw_index = wand_data<wand_data_raw>;
using wand_uniform_index = wand_data<wand_data_compressed<>>;
using wand_uniform_index_quantized = wand_data<wand_data_compressed<PayloadType::Quantized>>;


//string single_freq_path = "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms";
string single_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms";
string single_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/single_with_termscore/single_prefix";
string single_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/single_with_termscore/single_lexicon.txt";

//string duplet_freq_path = "/ssd2/home/bmmliu/logBaseFreq/2_term_freq_1.txt";
string duplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/duplet_cleaned.txt";
string duplet_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/duplet_with_termscore/duplet_prefix";
string duplet_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/duplet_with_termscore/duplet_lexicon.txt";

//string triplet_freq_path = "/ssd2/home/bmmliu/logBaseFreq/3_term_freq_1.txt";
string triplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/triplet_cleaned.txt";
string triplet_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/triplet_with_termscore/triplet_prefix";
string triplet_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/triplet_with_termscore/triplet_lexicon.txt";

//string quadruplet_freq_path = "/ssd2/home/bmmliu/logBaseFreq/4_term_freq_1.txt";
string quadruplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/quadruplet_cleaned.txt";
string quadruplet_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/quadruplet_with_termscore/quadruplet_prefix";
string quadruplet_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/quadruplet_with_termscore/quadruplet_lexicon.txt";

candidates::index_path single = std::make_tuple(single_gram_path, single_lexicon_path, single_prefix_path);
candidates::index_path duplet = std::make_tuple(duplet_gram_path, duplet_lexicon_path, duplet_prefix_path);
candidates::index_path triplet = std::make_tuple(triplet_gram_path, triplet_lexicon_path, triplet_prefix_path);
candidates::index_path quadruplet = std::make_tuple(quadruplet_gram_path, quadruplet_lexicon_path, quadruplet_prefix_path);

//candidates::realtime_heap_allocate allocate(single, duplet, triplet, quadruplet);

ifstream single_prefix_binary;
std::vector<char> single_prefix_binary_vec;

ifstream duplet_prefix_binary;

ifstream triplet_prefix_binary;

ifstream quadruplet_prefix_binary;

std::unique_ptr<pisa::Tokenizer> tokenizer = std::make_unique<EnglishTokenizer>();



struct Posting
{
    vector<uint32_t> term_ids;
    short total_score;
    short term_size;
    string term_string;
    int64_t cur_pos;
    int64_t end_pos;

    Posting(vector<uint32_t> term_ids, short total_score, short term_size, string term_string, int64_t cur_pos, int64_t end_pos)
        : term_ids(term_ids), total_score(total_score), term_size(term_size), term_string(term_string), cur_pos(cur_pos), end_pos(end_pos) {}
};

struct ComparePostingScore {
    bool operator()(Posting const& p1, Posting const& p2)
    {
        // return "true" if "p1" is ordered
        // before "p2", for example:
        return p1.total_score < p2.total_score;
    }
};

struct eqstr {
    bool operator()(const string &s1, const string &s2) const
    {
        return s1.compare(s2) == 0;
    }
};

struct eq64int
{
    bool operator()(const uint64_t &i1, const uint64_t &i2) const
    {
        return i1 == i2;
    }
};

struct eq32int
{
    bool operator()(const uint32_t &i1, const uint32_t &i2) const
    {
        return i1 == i2;
    }
};

std::set<uint32_t> parse_tuple(std::string const& line, size_t k)
{
    std::vector<std::string> term_ids;
    boost::algorithm::split(term_ids, line, boost::is_any_of(" \t"));
    if (term_ids.size() != k) {
        throw std::runtime_error(fmt::format(
            "Wrong number of terms in line: {} (expected {} but found {})", line, k, term_ids.size()));
    }

    std::set<uint32_t> term_ids_int;
    for (auto&& term_id: term_ids) {
        try {
            term_ids_int.insert(std::stoi(term_id));
        } catch (...) {
            throw std::runtime_error(
                fmt::format("Cannot convert {} to int in line: {}", term_id, line));
        }
    }
    return term_ids_int;
}

vector<string> split (const string &s, char delim) {
    vector<string> result;
    stringstream ss (s);
    string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}

vector<Query> make_exist_term_queries(std::string exist_term_file) {
    std::vector<::pisa::Query> q;
    std::string m_term_lexicon = "/home/bmmliu/data/cw09b/CW09B.fwd.termlex";
    auto parse_query = resolve_query_parser(q, std::make_unique<EnglishTokenizer>(), m_term_lexicon, std::nullopt, std::nullopt);
    std::ifstream is(exist_term_file);
    io::for_each_line(is, parse_query);
    return q;
}

vector<string> getAllPossibleComb(vector<uint32_t> &terms, int termConsidered) {
    vector<string> retVal_string;

    if (terms.size() >= 1 && termConsidered >= 1) {
        for (int i = 0; i < terms.size(); ++i) {
            retVal_string.push_back(to_string(terms[i]));
        }
    }
    if (terms.size() >= 2 && termConsidered >= 2) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                retVal_string.push_back(to_string(terms[i]) + "-" + to_string(terms[j]));
            }
        }
    }

    if (terms.size() >= 3 && termConsidered >= 3) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                for (int s = j + 1; s < terms.size(); ++s) {
                    retVal_string.push_back(to_string(terms[i]) + "-" + to_string(terms[j]) + "-" + to_string(terms[s]));
                }
            }
        }
    }

    if (terms.size() >= 4 && termConsidered >= 4) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                for (int s = j + 1; s < terms.size(); ++s) {
                    for (int t = s + 1; t < terms.size(); ++t) {
                        retVal_string.push_back(to_string(terms[i]) + "-" + to_string(terms[j]) + "-" + to_string(terms[s]) + "-" + to_string(terms[t]));
                    }
                }
            }
        }
    }

    return retVal_string;
}

vector<uint32_t> getTermsFromString (string &termStr) {
    vector<string> termsVecStr = split(termStr, '-');
    vector<uint32_t> terms;
    for (string str : termsVecStr) {
        terms.push_back(static_cast<uint32_t>(std::stoul(str)));
    }

    return terms;
}

string term_to_string (vector<uint32_t> &terms) {
    string terms_string = "";
    for (uint32_t term_id: terms) {
        terms_string += to_string(term_id) + "-";
    }
    terms_string.pop_back();

    return terms_string;
}

void load_lexicon(dense_hash_map<string, pair<int64_t, int64_t>, hash<string>, eqstr> &lex_map, string &freq_path, string &gram_path, string &lexicon_path, int gram_size) {
    vector<Query> exist_terms_queries = make_exist_term_queries(gram_path);
    vector<Query> freq_terms_queries = make_exist_term_queries(freq_path);
    unordered_set<string> freq_terms_string_set;

    for (auto const& query: freq_terms_queries) {
        vector<uint32_t> freq_query_terms = query.terms;
        string freq_term_id_string = term_to_string(freq_query_terms);
        freq_terms_string_set.insert(freq_term_id_string);
    }

    ifstream in_lex(lexicon_path);

    int count = 0;

    for (auto const& query: exist_terms_queries) {
        vector<uint32_t> unsorted_terms = query.terms;
        string term_id_string = term_to_string(unsorted_terms);

        if (gram_size != unsorted_terms.size() || freq_terms_string_set.find(term_id_string) == freq_terms_string_set.end()) {
            //clog << "loading non stemmed query" << endl;
            string lexline;
            std::getline(in_lex, lexline);
            continue;
        }

        string lexline;
        vector<string> lex_pair;

        std::getline(in_lex, lexline);
        boost::split(lex_pair, lexline, boost::is_any_of(" "), boost::token_compress_on);
        lex_map[term_id_string] = pair<int64_t, int64_t>(std::stoll(lex_pair[0], nullptr, 0), std::stoll(lex_pair[1], nullptr, 0));

        count++;
        if (count % 10000000 == 0) {
            clog << count << " " << gram_size << " term combine loaded" << endl;
        }
    }

    in_lex.close();
}

// return 1 if push successfully
int insert_posting_to_heap_vec(priority_queue<Posting, vector<Posting>, ComparePostingScore> &posting_max_heap, Posting cur_posting) {

    if (cur_posting.cur_pos < cur_posting.end_pos) {
        int did;
        short total_score = 0;

        if (cur_posting.term_size == 1) {

            std::memcpy(&did, single_prefix_binary_vec.data() + cur_posting.cur_pos, sizeof(int));
            for (int i = 0; i < cur_posting.term_size; i++) {
                unsigned char score;
                std::memcpy(&score, single_prefix_binary_vec.data() + cur_posting.cur_pos + sizeof(int) + i, sizeof(unsigned char));
                total_score += static_cast<short>(score);
            }

        } else {
            cerr << "Wrong posting was considered" << endl;
            return 0;
        }

        cur_posting.total_score = total_score;
        cur_posting.cur_pos += 4 + cur_posting.term_size;
        posting_max_heap.push(cur_posting);
        return 1;
    }
    return 0;
}

int insert_posting_to_heap_vec_chunk(priority_queue<Posting, vector<Posting>, ComparePostingScore> &posting_max_heap, Posting cur_posting, int chunk_size, dense_hash_map<uint64_t, pair<bitset<MaxQueryLen>, short>, hash<uint64_t>, eq64int> &did_scores_map) {

    if (cur_posting.cur_pos < cur_posting.end_pos) {
        int did;
        short total_score = 0;

        if (cur_posting.term_size == 1) {

            std::memcpy(&did, single_prefix_binary_vec.data() + cur_posting.cur_pos, sizeof(int));
            for (int i = 0; i < cur_posting.term_size; i++) {
                unsigned char score;
                std::memcpy(&score, single_prefix_binary_vec.data() + cur_posting.cur_pos + sizeof(int) + i, sizeof(unsigned char));
                total_score += static_cast<short>(score);
            }

        } else {
            cerr << "Wrong posting was considered" << endl;
            return 0;
        }

        cur_posting.total_score = total_score;
        cur_posting.cur_pos += 4 + cur_posting.term_size;
        posting_max_heap.push(cur_posting);
        return 1;
    }
    return 0;
}

// return 1 if push successfully
int insert_posting_to_heap(priority_queue<Posting, vector<Posting>, ComparePostingScore> &posting_max_heap, Posting cur_posting, int chunk_size, dense_hash_map<uint64_t, pair<bitset<MaxQueryLen>, short>, hash<uint64_t>, eq64int> &did_scores_map, dense_hash_map<uint32_t, short, hash<uint32_t>, eq32int> &term_position_map) {

    int count = 0;
    vector<int> didVec;
    vector<vector<short>> scoresVec;
    vector<short> totalScoreVec;

    while (cur_posting.cur_pos < cur_posting.end_pos && count < chunk_size) {
        int did;
        vector<short> scores;
        short total_score = 0;
        if (cur_posting.term_size == 1) {
            single_prefix_binary.seekg(cur_posting.cur_pos);
            single_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < cur_posting.term_size; i++) {
                unsigned char score;
                single_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                scores.push_back((short)score);
                total_score += (short)score;
            }
        } else if (cur_posting.term_size == 2) {
            duplet_prefix_binary.seekg(cur_posting.cur_pos);
            duplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < cur_posting.term_size; i++) {
                unsigned char score;
                duplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                scores.push_back((short)score);
                total_score += (short)score;
            }
        } else if (cur_posting.term_size == 3) {
            triplet_prefix_binary.seekg(cur_posting.cur_pos);
            triplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < cur_posting.term_size; i++) {
                unsigned char score;
                triplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                scores.push_back((short)score);
                total_score += (short)score;
            }
        } else if (cur_posting.term_size == 4) {
            quadruplet_prefix_binary.seekg(cur_posting.cur_pos);
            quadruplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < cur_posting.term_size; i++) {
                unsigned char score;
                quadruplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                scores.push_back((short)score);
                total_score += (short)score;
            }
        } else {
            cerr << "Wrong posting was considered" << endl;
            continue;
        }

        cur_posting.cur_pos += 4 + cur_posting.term_size;
        count++;

        didVec.push_back(did);
        scoresVec.push_back(scores);
        // warning: should be next one ?
        totalScoreVec.push_back(total_score);
    }

    if (didVec.size() != scoresVec.size() || scoresVec.size() != totalScoreVec.size()) {
        cout << "warning" << didVec.size() << scoresVec.size() << totalScoreVec.size() << endl;
    }

    if (didVec.size() == 0 || scoresVec.size() == 0 || totalScoreVec.size() == 0) {
        return count;
    }

    for (int i = 0; i < didVec.size(); i++) {
        auto& did_scores = did_scores_map[didVec[i]];

        if (did_scores.first.count() == cur_posting.term_size) {
            continue;
        }
        for (int j = 0; j < cur_posting.term_size; j++) {
            uint32_t cur_term_id = cur_posting.term_ids[j];

            short term_position = term_position_map[cur_term_id];
            if (!did_scores.first.test(term_position)) {
                did_scores.second += scoresVec[i][j];
                did_scores.first.set(term_position);
            }
        }
    }

    cur_posting.total_score = totalScoreVec.back();
    if (cur_posting.cur_pos < cur_posting.end_pos) {
        posting_max_heap.push(cur_posting);
    }

    return count;
}

template <typename IndexType, typename WandType>
void kt_thresholds(
    const std::string& index_filename,
    const std::string& wand_data_filename,
    const std::vector<Query>& queries,
    std::string const& type,
    ScorerParams const& scorer_params,
    uint64_t k,
    bool quantized,
    std::optional<std::string> pairs_filename,
    std::optional<std::string> triples_filename,
    std::optional<std::string> exist_term_filename,
    bool all_pairs,
    bool all_triples)
{
    string index_path = "/home/jg6226/data/Hit_Ratio_Project/Uncompressed_Index_Quantized";
    string index_name = "CW09B.quantized.url.inv";
    string term_lexicon_path = "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms";
    string index_lexicon_path = index_path + "/CW09B.quantized.url.inv.indexlex";
    string blocklast_file_path = index_path + "/CW09B.quantized.url.inv.blocklast";


    IndexType index;
    mio::mmap_source m(index_filename.c_str());
    mapper::map(index, m);

    WandType wdata;

    auto scorer = scorer::from_params(scorer_params, wdata);

    mio::mmap_source md;
    std::error_code error;
    md.map(wand_data_filename, error);
    if (error) {
        spdlog::error("error mapping file: {}, exiting...", error.message());
        std::abort();
    }
    mapper::map(wdata, md, mapper::map_flags::warmup);

    using Pair = std::set<uint32_t>;
    std::unordered_set<Pair, boost::hash<Pair>> pairs_set;

    using Triple = std::set<uint32_t>;
    std::unordered_set<Triple, boost::hash<Triple>> triples_set;

    std::string line;
    if (all_pairs) {
        spdlog::info("All pairs are available.");
    }
    if (pairs_filename) {
        std::ifstream pin(*pairs_filename);
        while (std::getline(pin, line)) {
            pairs_set.insert(parse_tuple(line, 2));
        }
        spdlog::info("Number of pairs loaded: {}", pairs_set.size());
    }

    if (all_triples) {
        spdlog::info("All triples are available.");
    }
    if (triples_filename) {
        std::ifstream trin(*triples_filename);
        while (std::getline(trin, line)) {
            triples_set.insert(parse_tuple(line, 3));
        }
        spdlog::info("Number of triples loaded: {}", triples_set.size());
    }

    // get all kinds of combinations
    //int numberOfTerms = 3;
    //int d = k * 10;
    int getDandTFromFileNameFlag = 1;
    vector<string> argStr;

    if (exist_term_filename && getDandTFromFileNameFlag) {
        argStr = split(*exist_term_filename, ',');
    }

    int budget = atoi(argStr[0].c_str());
    int termConsidered = 4;

    string freq = argStr[1].c_str();
    string sin_freq_file = "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms";
    string dup_freq_file = "/ssd2/home/bmmliu/logBaseFreq/2_term_freq_" + string(1, freq[1]) + ".txt";
    string tri_freq_file = "/ssd2/home/bmmliu/logBaseFreq/3_term_freq_" + string(1, freq[2]) + ".txt";
    string qud_freq_file = "/ssd2/home/bmmliu/logBaseFreq/4_term_freq_" + string(1, freq[3]) + ".txt";


    cout << "k = " << k << endl;
    cout << "budget = " << budget << endl;

    dense_hash_map<string, pair<int64_t, int64_t>, hash<string>, eqstr> lex_map;
    lex_map.set_empty_key("NULL");

    single_prefix_binary.open(single_prefix_path, std::ios::in | std::ios::binary);
    /*single_prefix_binary.seekg(0, std::ios::end);
    std::streamsize fileSize = single_prefix_binary.tellg();
    single_prefix_binary.seekg(0, std::ios::beg);

    single_prefix_binary_vec.resize(fileSize);
    single_prefix_binary.read(single_prefix_binary_vec.data(), fileSize);*/
    load_lexicon(lex_map, sin_freq_file, single_gram_path, single_lexicon_path, 1);

    if (termConsidered >= 2) {
        load_lexicon(lex_map, dup_freq_file, duplet_gram_path, duplet_lexicon_path, 2);
        duplet_prefix_binary.open(duplet_prefix_path, std::ios::in | std::ios::binary);
    }
    if (termConsidered >= 3) {
        load_lexicon(lex_map, tri_freq_file, triplet_gram_path, triplet_lexicon_path, 3);
        triplet_prefix_binary.open(triplet_prefix_path, std::ios::in | std::ios::binary);
    }
    if (termConsidered >= 4) {
        load_lexicon(lex_map, qud_freq_file, quadruplet_gram_path, quadruplet_lexicon_path, 4);
        quadruplet_prefix_binary.open(quadruplet_prefix_path, std::ios::in | std::ios::binary);
    }




    vector<float> realThreshold;

    vector<float> beforeLookupThreshold;
    vector<float> finalThreshold;
    vector<int> heapBudgetUsed;
    vector<float> heapTimeVec;
    vector<float> lookupTimeVec;
    vector<float> wholeTimeVec;

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end = std::chrono::high_resolution_clock::now();

    int count = 0;
    int line_num = 0;

    for (auto const& query: queries) {
        line_num++;

        auto terms = query.terms;
        const int query_length = terms.size();

        topk_queue topk_old(1000);
        wand_query wand_q_old(topk_old);
        // calculate all terms threshold
        wand_q_old(make_max_scored_cursors(index, wdata, *scorer, query), index.num_docs());
        topk_old.finalize();
        auto allTermResults = topk_old.topk();

        topk_old.clear();
        float allTermThreshold = -1.0;
        if (allTermResults.size() >= k) {
            allTermThreshold = allTermResults[k - 1].first;
        }

        realThreshold.push_back(allTermThreshold);

        // If doc less than k
        if (allTermThreshold == -1.0 || query_length > MaxQueryLen) {
            finalThreshold.push_back(0.0);
            beforeLookupThreshold.push_back(0.0);
            wholeTimeVec.push_back(0.0);
            heapTimeVec.push_back(0.0);
            lookupTimeVec.push_back(0.0);
            heapBudgetUsed.push_back(0);
            continue;
        }

        t_start = std::chrono::high_resolution_clock::now();
        priority_queue<Posting, vector<Posting>, ComparePostingScore> posting_max_heap;
        vector<string> allPossibleComb = getAllPossibleComb(terms, termConsidered);
        float curEstimate;

        // Last one will be the total score
        dense_hash_map<uint64_t, pair<bitset<MaxQueryLen>, short>, hash<uint64_t>, eq64int> did_scores_map;
        did_scores_map.set_empty_key(numeric_limits<uint64_t>::max());
        did_scores_map.resize(budget + 1);

        dense_hash_map<uint32_t, short, hash<uint32_t>, eq32int> term_position_map;
        term_position_map.set_empty_key(numeric_limits<uint32_t>::max());

        short pos = 0;
        for (uint32_t term : terms) {
            term_position_map[term] = pos;
            pos++;
        }

        for (string comb : allPossibleComb) {
            if (lex_map.find(comb) != lex_map.end()) {
                //insert_posting_to_heap(posting_max_heap, comb, 0, lex_map);
                //insert_posting_to_heap_map(posting_max_heap, comb, 0, term_score_map);
                vector<uint32_t> term_ids = getTermsFromString(comb);
                short term_size = term_ids.size();
                int64_t start_pos = lex_map[comb].first;
                int64_t end_pos = lex_map[comb].second;
                int64_t cur_pos = start_pos + 4;

                if (cur_pos < end_pos) {
                    int did;
                    short total_score = 0;

                    if (term_size == 1) {
                        single_prefix_binary.seekg(cur_pos);
                        single_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
                        for (int i = 0; i < term_size; i++) {
                            unsigned char score;
                            single_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                            total_score += (short)score;
                        }
                    } else if (term_size == 2) {
                        duplet_prefix_binary.seekg(cur_pos);
                        duplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
                        for (int i = 0; i < term_size; i++) {
                            unsigned char score;
                            duplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                            total_score += (short)score;
                        }
                    } else if (term_size == 3) {
                        triplet_prefix_binary.seekg(cur_pos);
                        triplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
                        for (int i = 0; i < term_size; i++) {
                            unsigned char score;
                            triplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                            total_score += (short)score;
                        }
                    } else if (term_size == 4) {
                        quadruplet_prefix_binary.seekg(cur_pos);
                        quadruplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
                        for (int i = 0; i < term_size; i++) {
                            unsigned char score;
                            quadruplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                            total_score += (short)score;
                        }
                    } else {
                        cerr << "Wrong posting was considered" << endl;
                        continue;
                    }

                    Posting cur_posting = Posting(term_ids, total_score, term_size, comb, cur_pos, end_pos);
                    posting_max_heap.push(cur_posting);
                }
            }
        }

        //clog << "initial heap size is: " << posting_max_heap.size() << endl;
        int cur_budget = budget;
        auto t_heap_start = std::chrono::high_resolution_clock::now();

        while (cur_budget > 0 && posting_max_heap.size() > 0) {
            Posting top_posting = posting_max_heap.top();
            posting_max_heap.pop();
            int if_push_success = insert_posting_to_heap(posting_max_heap, top_posting, 1, did_scores_map, term_position_map);

            if (if_push_success > 0) {
                cur_budget = cur_budget - if_push_success;
            }

            /*auto& did_scores = did_scores_map[top_posting.did];

            if (did_scores.first.count() == query_length) {
                continue;
            }
            for (int i = 0; i < top_posting.term_ids.size(); i++) {
                uint32_t cur_term_id = top_posting.term_ids[i];
                short cur_term_score = top_posting.scores[i];
                short term_position = term_position_map[cur_term_id];
                if (!did_scores.first.test(term_position)) {
                    did_scores.second += cur_term_score;
                    did_scores.first.set(term_position);
                }
            }*/
            //clog << "did scores map count is: " << did_scores_map.size() << endl;
        }

        auto t_heap_end = std::chrono::high_resolution_clock::now();
        heapTimeVec.push_back(std::chrono::duration<double, std::nano>(t_heap_end-t_heap_start).count());
        heapBudgetUsed.push_back(budget - cur_budget);

        // Document MUF Before LookUp
        if (did_scores_map.size() >= k) {
            vector<short> all_combined_scores;

            for(auto key_scores : did_scores_map) {
                all_combined_scores.push_back(key_scores.second.second);
            }

            sort(all_combined_scores.begin(), all_combined_scores.end(), greater<short>());
            beforeLookupThreshold.push_back(all_combined_scores[k - 1]);
        } else {
            beforeLookupThreshold.push_back(0.0);
        }


        // Do lookup here
        auto t_lookup_start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < query_length; i++) {
            vector<uint64_t> curDidForLookUp;

            for(auto key_scores : did_scores_map) {
                if (!key_scores.second.first.test(i)) {
                    curDidForLookUp.push_back(key_scores.first);
                }
            }

            sort(curDidForLookUp.begin(), curDidForLookUp.end());

            uint32_t curTerm = terms[i];
            Query single_term_query = query;
            single_term_query.terms = {curTerm};


            vector<double> lookupResult(curDidForLookUp.size(), 0.0);

            vector<double> apiLookupResult;
            vector<uint64_t> curDidForLookUpCopy(curDidForLookUp);

            if (curDidForLookUp.size() > 0) {

                //apiLookupResult = allocate.did_score_list_final<IndexType, WandType>(index, wdata, scorer, single_term_query, curDidForLookUpCopy);
                apiLookupResult.resize(curDidForLookUp.size(), 0.0);
                if (apiLookupResult.size() != curDidForLookUp.size()) {
                    continue;
                }

                for (int apiLookupResultIndex = 0; apiLookupResultIndex < apiLookupResult.size(); apiLookupResultIndex++) {
                    lookupResult[apiLookupResultIndex] += apiLookupResult[apiLookupResultIndex];
                }
            }

            for (int j = 0; j < curDidForLookUp.size(); j++) {
                if (static_cast<short>(lookupResult[j]) <= 0) {
                    continue;
                }
                did_scores_map[curDidForLookUp[j]].first.set(i);
                did_scores_map[curDidForLookUp[j]].second += static_cast<short>(lookupResult[j]);
            }
        }

        auto t_lookup_end = std::chrono::high_resolution_clock::now();
        lookupTimeVec.push_back(std::chrono::duration<double, std::nano>(t_lookup_end-t_lookup_start).count());

        if (did_scores_map.size() >= k) {
            vector<short> all_combined_scores;

            for(auto key_scores : did_scores_map) {
                all_combined_scores.push_back(key_scores.second.second);
            }

            sort(all_combined_scores.begin(), all_combined_scores.end(), greater<short>());
            curEstimate = all_combined_scores[k - 1];
        } else {
            curEstimate = 0.0;
        }


        t_end = std::chrono::high_resolution_clock::now();
        finalThreshold.push_back(curEstimate);
        wholeTimeVec.push_back(std::chrono::duration<double, std::nano>(t_end-t_start).count());

        count++;
        if (count % 100 == 0) {
            clog << count << "queries consider terms = " << termConsidered << " k = " << k << " processed -- combine terms" << endl;
        }
    }

    single_prefix_binary.close();
    duplet_prefix_binary.close();
    triplet_prefix_binary.close();
    quadruplet_prefix_binary.close();

    // calculate Before LookUp MUF
    if (realThreshold.size() == beforeLookupThreshold.size()) {
        double TotalMUF = 0.0;
        for(int i = 0; i < realThreshold.size(); i++) {
            TotalMUF += (beforeLookupThreshold[i] / realThreshold[i]);
        }

        cout << beforeLookupThreshold.size() << " values considered, MUF Before Lookup for " << beforeLookupThreshold.size() << " queries is: " << (TotalMUF / beforeLookupThreshold.size()) << endl;
    }

    // calculate Final MUF
    if (realThreshold.size() == finalThreshold.size()) {
        double TotalMUF = 0.0;
        for(int i = 0; i < realThreshold.size(); i++) {
            TotalMUF += (finalThreshold[i] / realThreshold[i]);
        }

        cout << finalThreshold.size() << " values considered, MUF Final for " << finalThreshold.size() << " queries is: " << (TotalMUF / finalThreshold.size()) << endl;
    }

    // calculate Heap Budget Used
    int TotalHeapBudgetUsed = 0;
    for(int i = 0; i < heapBudgetUsed.size(); i++) {
        TotalHeapBudgetUsed += heapBudgetUsed[i];
    }

    cout << heapBudgetUsed.size() << " values considered, Average Heap Budget Used for " << heapBudgetUsed.size() << " queries is: " << (TotalHeapBudgetUsed / heapBudgetUsed.size()) << endl;

    // calculate Heap Time Used
    double TotalHeapTimeUsed = 0.0;
    for(int i = 0; i < heapTimeVec.size(); i++) {
        TotalHeapTimeUsed += heapTimeVec[i];
    }

    cout << heapTimeVec.size() << " values considered, Average Heap Time Used for " << heapTimeVec.size() << " queries is: " << (TotalHeapTimeUsed / heapTimeVec.size()) << endl;
    cout << "Average Heap Time Used for each Posting is " << (TotalHeapTimeUsed / heapTimeVec.size()/(TotalHeapBudgetUsed / heapBudgetUsed.size())) << endl;

    // calculate Lookup Time Used
    double TotalLookupTimeUsed = 0.0;
    for(int i = 0; i < lookupTimeVec.size(); i++) {
        TotalLookupTimeUsed += lookupTimeVec[i];
    }

    cout << lookupTimeVec.size() << " values considered, Average Lookup Time Used for " << lookupTimeVec.size() << " queries is: " << (TotalLookupTimeUsed / lookupTimeVec.size()) << endl;

    // calculate Whole Time Used
    double WholeTimeUsed = 0.0;
    for(int i = 0; i < wholeTimeVec.size(); i++) {
        WholeTimeUsed += wholeTimeVec[i];
    }

    cout << wholeTimeVec.size() << " values considered, Average all processing Time Used for " << wholeTimeVec.size() << " queries is: " << (WholeTimeUsed / wholeTimeVec.size()) << endl;

    // print result
    /*std::cout << "real threshold" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < realThreshold.size(); i++) {
        cout << realThreshold[i] << '\n';
    }

    std::cout << "single threshold" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < finalThreshold.size(); i++) {
        cout << finalThreshold[i] << '\n';
    }
    std::cout << "single estimated K" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < singleEstimatedK.size(); i++) {
        cout << singleEstimatedK[i] << '\n';
    }
    std::cout << "single estimated time" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < wholeTimeVec.size(); i++) {
        cout << wholeTimeVec[i] << '\n';
    }

    std::cout << "single budget used" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < heapBudgetUsed.size(); i++) {
        cout << heapBudgetUsed[i] << '\n';
    }*/
}

int main(int argc, const char** argv) {

    std::string documents_file;

    std::string query_filename;
    std::optional<std::string> pairs_filename;
    std::optional<std::string> triples_filename;
    std::optional<std::string> exist_term_filename;
    std::string index_filename;
    std::string wand_data_filename;
    bool quantized = false;

    bool all_pairs = false;
    bool all_triples = false;

    App<arg::Index, arg::WandData<arg::WandMode::Required>, arg::Query<arg::QueryMode::Ranked>, arg::Scorer>
        app{"A tool for performing threshold estimation using the k-highest impact score for each "
            "term, pair or triple of a query. Pairs and triples are only used if provided with "
            "--pairs and --triples respectively."};
    auto pairs = app.add_option(
        "-p,--pairs", pairs_filename, "A tab separated file containing all the cached term pairs");
    auto triples = app.add_option(
        "-t,--triples",
        triples_filename,
        "A tab separated file containing all the cached term triples");
    auto exist_terms = app.add_option(
        "--exist", exist_term_filename, "A newline separated file containing all the exist term");

    app.add_flag("--all-pairs", all_pairs, "Consider all term pairs of a query")->excludes(pairs);
    app.add_flag("--all-triples", all_triples, "Consider all term triples of a query")->excludes(triples);
    app.add_flag("--quantized", quantized, "Quantizes the scores");

    CLI11_PARSE(app, argc, argv);

    auto params = std::make_tuple(
        app.index_filename(),
        app.wand_data_path(),
        app.queries(),
        app.index_encoding(),
        app.scorer_params(),
        app.k(),
        quantized,
        pairs_filename,
        triples_filename,
        exist_term_filename,
        all_pairs,
        all_triples);

    /**/
    if (false) {
#define LOOP_BODY(R, DATA, T)                                                                      \
    }                                                                                              \
    else if (app.index_encoding() == BOOST_PP_STRINGIZE(T))                                        \
    {                                                                                              \
        if (app.is_wand_compressed()) {                                                            \
            if (quantized) {                                                                       \
                std::apply(                                                                        \
                    kt_thresholds<BOOST_PP_CAT(T, _index), wand_uniform_index_quantized>, params); \
            } else {                                                                               \
                std::apply(kt_thresholds<BOOST_PP_CAT(T, _index), wand_uniform_index>, params);    \
            }                                                                                      \
        } else {                                                                                   \
            std::apply(kt_thresholds<BOOST_PP_CAT(T, _index), wand_raw_index>, params);            \
        }
        /**/
        BOOST_PP_SEQ_FOR_EACH(LOOP_BODY, _, PISA_INDEX_TYPES);
#undef LOOP_BODY

    } else {
        spdlog::error("Unknown type {}", app.index_encoding());
    }
}
