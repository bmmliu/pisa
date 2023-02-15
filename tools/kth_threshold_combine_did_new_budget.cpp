#include <iostream>
#include <optional>
#include <unordered_set>
#include <queue>
#include <chrono>
#include <fstream>
#include <sstream>
#include <sparsehash/dense_hash_set>
#include <sparsehash/dense_hash_map>

#include "boost/algorithm/string/classification.hpp"
#include "boost/algorithm/string/split.hpp"
#include <boost/functional/hash.hpp>

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

using namespace pisa;
using term_id_type = uint32_t;
//using ext::hash;
using google::dense_hash_set;
using google::dense_hash_map;

string single_freq_path = "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms";
string single_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms";
string single_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/single_with_termscore/single_prefix";
string single_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/single_with_termscore/single_lexicon.txt";

string duplet_freq_path = "/ssd2/home/bmmliu/logBaseFreq/2_term_freq_2.txt";
string duplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/duplet_cleaned.txt";
string duplet_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/duplet_with_termscore/duplet_prefix";
string duplet_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/duplet_with_termscore/duplet_lexicon.txt";

string triplet_freq_path = "/ssd2/home/bmmliu/logBaseFreq/3_term_freq_2.txt";
string triplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/triplet_cleaned.txt";
string triplet_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/triplet_with_termscore/triplet_prefix";
string triplet_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/triplet_with_termscore/triplet_lexicon.txt";

string quadruplet_freq_path = "/ssd2/home/bmmliu/logBaseFreq/4_term_freq_2.txt";
string quadruplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/quadruplet_cleaned.txt";
string quadruplet_prefix_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/quadruplet_with_termscore/quadruplet_prefix";
string quadruplet_lexicon_path = "/ssd3/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/First_Layer_Index/quadruplet_with_termscore/quadruplet_lexicon.txt";

ifstream single_prefix_binary;

ifstream duplet_prefix_binary;

ifstream triplet_prefix_binary;

ifstream quadruplet_prefix_binary;

struct Posting
{
    uint64_t did;
    short score;
    string term_string;
    int rank;

    Posting(uint64_t did, short score, string term_string, int rank)
        : did(did), score(score), term_string(term_string), rank(rank) {}
};

struct ComparePostingScore {
    bool operator()(Posting const& p1, Posting const& p2)
    {
        // return "true" if "p1" is ordered
        // before "p2", for example:
        return p1.score < p2.score;
    }
};

struct eqstr {
    bool operator()(const string &s1, const string &s2) const
    {
        return s1.compare(s2) == 0;
    }
};

struct eqint
{
    bool operator()(const uint64_t &i1, const uint64_t &i2) const
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
    auto parse_query = resolve_query_parser(q, m_term_lexicon, std::nullopt, std::nullopt);
    std::ifstream is(exist_term_file);
    io::for_each_line(is, parse_query);
    return q;
}

bool cmpDidTPair(pair<uint64_t, short>& a, pair<uint64_t, short>& b){
    return a.second > b.second;
}

bool ifDupTerm(vector<uint32_t> terms) {
    unordered_set<uint32_t> memo;
    for (uint32_t term : terms) {
        if (memo.find(term) != memo.end()) {
            return true;
        }
        memo.insert(term);
    }

    return false;
}


vector<string> getAllPossibleComb(vector<uint32_t> &terms, int termConsidered) {
    vector<vector<int>> retVal;
    vector<string> retVal_string;

    if (terms.size() >= 1 && termConsidered >= 1) {
        for (int i = 0; i < terms.size(); ++i) {
            retVal.push_back({i});
        }
    }
    if (terms.size() >= 2 && termConsidered >= 2) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                retVal.push_back({i, j});
            }
        }
    }

    if (terms.size() >= 3 && termConsidered >= 3) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                for (int s = j + 1; s < terms.size(); ++s) {
                    retVal.push_back({i, j, s});
                }
            }
        }
    }

    if (terms.size() >= 4 && termConsidered >= 4) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                for (int s = j + 1; s < terms.size(); ++s) {
                    for (int t = s + 1; t < terms.size(); ++t) {
                        retVal.push_back({i, j, s, t});
                    }
                }
            }
        }
    }


    for (vector<int> comb : retVal) {
        string CombStr = "";
        for (int singleTerm : comb) {
            CombStr += to_string(terms[singleTerm]) + "-";
        }
        CombStr.pop_back();
        retVal_string.push_back(CombStr);
    }

    return retVal_string;
}

vector<uint32_t> getTermsFromString (string termStr) {
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
        if (count % 1000000 == 0) {
            clog << count << " " << gram_size << " term combine loaded" << endl;
        }
    }

    in_lex.close();
}

// return 1 if push successfully
int insert_posting_to_heap(priority_queue<Posting, vector<Posting>, ComparePostingScore> &posting_max_heap, string &term_string, int rank, dense_hash_map<string, pair<int64_t, int64_t>, hash<string>, eqstr> &lex_map) {
    vector<uint32_t> terms = getTermsFromString(term_string);
    int64_t start_pos = lex_map[term_string].first;
    int64_t end_pos = lex_map[term_string].second;
    int64_t cur_pos = start_pos + 4;
    cur_pos = cur_pos + rank * (4 + terms.size());

    if (cur_pos < end_pos) {
        int did;
        short total_score = 0;
        if (terms.size() == 1) {
            single_prefix_binary.seekg(cur_pos);
            single_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < terms.size(); i++) {
                unsigned char score;
                single_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                total_score += (short)score;
            }
        } else if (terms.size() == 2) {
            duplet_prefix_binary.seekg(cur_pos);
            duplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < terms.size(); i++) {
                unsigned char score;
                duplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                total_score += (short)score;
            }
        } else if (terms.size() == 3) {
            triplet_prefix_binary.seekg(cur_pos);
            triplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < terms.size(); i++) {
                unsigned char score;
                triplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                total_score += (short)score;
            }
        } else if (terms.size() == 4) {
            quadruplet_prefix_binary.seekg(cur_pos);
            quadruplet_prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < terms.size(); i++) {
                unsigned char score;
                quadruplet_prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                total_score += (short)score;
            }
        } else {
            cerr << "Wrong posting was considered" << endl;
        }

        posting_max_heap.push(Posting(did, total_score, term_string, rank));
        return 1;
    }
    return 0;
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
    int termConsidered = atoi(argStr[1].c_str());
    //int d = k * atoi(argStr[2].c_str());

    dense_hash_map<string, pair<int64_t, int64_t>, hash<string>, eqstr> lex_map;
    lex_map.set_empty_key("NULL");

    load_lexicon(lex_map, single_freq_path, single_gram_path, single_lexicon_path, 1);
    if (termConsidered >= 2) {
        load_lexicon(lex_map, duplet_freq_path, duplet_gram_path, duplet_lexicon_path, 2);
    }
    if (termConsidered >= 3) {
        load_lexicon(lex_map, triplet_freq_path, triplet_gram_path, triplet_lexicon_path, 3);
    }
    if (termConsidered >= 4) {
        load_lexicon(lex_map, quadruplet_freq_path, quadruplet_gram_path, quadruplet_lexicon_path, 4);
    }

    single_prefix_binary.open(single_prefix_path, std::ios::in | std::ios::binary);
    duplet_prefix_binary.open(duplet_prefix_path, std::ios::in | std::ios::binary);
    triplet_prefix_binary.open(triplet_prefix_path, std::ios::in | std::ios::binary);
    quadruplet_prefix_binary.open(quadruplet_prefix_path, std::ios::in | std::ios::binary);


    vector<float> realThreshold;

    vector<float> singleThreshold;
    vector<int> singleEstimatedK;
    vector<int> budgetUsed;
    vector<float> singleQueryTimeVec;

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end = std::chrono::high_resolution_clock::now();

    int count = 0;
    for (auto const& query: queries) {

        auto terms = query.terms;

        topk_queue topk_old(k * 1000);
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
        if (allTermThreshold == -1.0) {
            singleThreshold.push_back(-1.0);
            singleEstimatedK.push_back(-1);
            singleQueryTimeVec.push_back(-1.0);
            budgetUsed.push_back(-1);
            continue;
        }

        t_start = std::chrono::high_resolution_clock::now();
        priority_queue<Posting, vector<Posting>, ComparePostingScore> posting_max_heap;
        vector<string> allPossibleComb = getAllPossibleComb(terms, termConsidered);
        float curEstimate;


        dense_hash_set<uint64_t, hash<uint64_t>, eqint> popped_did;
        popped_did.set_empty_key(NULL);

        //unordered_set<uint64_t> popped_did;

        for (string comb : allPossibleComb) {
            if (lex_map.find(comb) != lex_map.end()) {
                vector<uint32_t> terms = getTermsFromString(comb);
                insert_posting_to_heap(posting_max_heap, comb, 0, lex_map);
            }
        }

        //clog << "initial heap size is: " << posting_max_heap.size() << endl;

        Posting top_posting(0, -1, "wrong", -1);
        int cur_budget = budget;

        while (popped_did.size() < k && cur_budget > 0 && posting_max_heap.size() > 0) {
            top_posting = posting_max_heap.top();
            posting_max_heap.pop();
            int if_push_success = insert_posting_to_heap(posting_max_heap, top_posting.term_string, top_posting.rank + 1, lex_map);

            if (if_push_success == 1) {
                //clog << "push success" << endl;
                cur_budget--;
            }
            popped_did.insert(top_posting.did);
            //clog << "popped did count is: " << popped_did.size() << endl;
        }


        if (popped_did.size() == k) {
            curEstimate = (double)top_posting.score;
            budgetUsed.push_back(budget - cur_budget);
            //clog << "budget remaining: " << cur_budget << endl;
        } else {
            curEstimate = -2.0;
        }

        t_end = std::chrono::high_resolution_clock::now();
        singleThreshold.push_back(curEstimate);
        singleQueryTimeVec.push_back(std::chrono::duration<double, std::milli>(t_end-t_start).count());

        if (curEstimate < 0) {
            singleEstimatedK.push_back(-2);
            budgetUsed.push_back(-2);
            continue;
        }

        for (int i = 0; i < allTermResults.size() - 1; i++) {
            if (allTermResults[i].first >= curEstimate && allTermResults[i + 1].first <= curEstimate) {
                singleEstimatedK.push_back(i + 2);
                break;
            } else if (i == allTermResults.size() - 2) {
                singleEstimatedK.push_back(i + 2);
                break;
            }
        }

        count++;
        if (count % 10 == 0) {
            clog << count << "queries consider terms = " << termConsidered << " k = " << k << " processed -- combine did" << endl;
        }
    }

    single_prefix_binary.close();
    duplet_prefix_binary.close();
    triplet_prefix_binary.close();
    quadruplet_prefix_binary.close();

    // print result
    std::cout << "real threshold" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < realThreshold.size(); i++) {
        cout << realThreshold[i] << '\n';
    }

    std::cout << "single threshold" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < singleThreshold.size(); i++) {
        cout << singleThreshold[i] << '\n';
    }
    std::cout << "single estimated K" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < singleEstimatedK.size(); i++) {
        cout << singleEstimatedK[i] << '\n';
    }
    std::cout << "single estimated time" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < singleQueryTimeVec.size(); i++) {
        cout << singleQueryTimeVec[i] << '\n';
    }

    std::cout << "single budget used" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < budgetUsed.size(); i++) {
        cout << budgetUsed[i] << '\n';
    }

}

using wand_raw_index = wand_data<wand_data_raw>;
using wand_uniform_index = wand_data<wand_data_compressed<>>;
using wand_uniform_index_quantized = wand_data<wand_data_compressed<PayloadType::Quantized>>;

int main(int argc, const char** argv)
{
    spdlog::drop("");
    spdlog::set_default_logger(spdlog::stderr_color_mt(""));

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
