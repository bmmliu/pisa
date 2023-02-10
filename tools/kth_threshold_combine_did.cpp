#include <iostream>
#include <optional>
#include <unordered_set>
#include <queue>
#include <chrono>
#include <fstream>
#include <sstream>

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

bool cmpDidTPair(pair<uint64_t, float>& a, pair<uint64_t, float>& b){
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

void load_exist_terms(unordered_set<string> &cached_terms, string cache_filename) {
    vector<Query> exist_terms_queries = make_exist_term_queries(cache_filename);
    for (auto const& query: exist_terms_queries) {
        // Get sorted query and convert it to string

        if (query.terms.size() == 0) {
            continue;
        }
        vector<uint32_t> sorted_terms = query.terms;
        string sorted_terms_string = "";

        sort(sorted_terms.begin(), sorted_terms.end());
        for (uint32_t term_id: sorted_terms) {
            sorted_terms_string += to_string(term_id) + "-";
        }
        sorted_terms_string.pop_back();
        cached_terms.insert(sorted_terms_string);
    }
}

vector<vector<string>> getAllPossibleSlicing(vector<uint32_t> &terms, int termConsidered) {
    sort(terms.begin(), terms.end());
    vector<vector<vector<int>>> retVal;
    vector<vector<string>> retVal_string;

    if (terms.size() >= 1 && termConsidered >= 1) {
        for (int i = 0; i < terms.size(); ++i) {
            retVal.push_back({{i}});
        }
    }
    if (terms.size() >= 2 && termConsidered >= 2) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                retVal.push_back({{i, j}});
            }
        }
    }

    if (terms.size() >= 3 && termConsidered >= 3) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                for (int s = j + 1; s < terms.size(); ++s) {
                    retVal.push_back({{i, j, s}});
                }
            }
        }
    }

    if (terms.size() >= 4 && termConsidered >= 4) {
        for (int i = 0; i < terms.size(); ++i) {
            for (int j = i + 1; j < terms.size(); ++j) {
                for (int s = j + 1; s < terms.size(); ++s) {
                    for (int t = s + 1; t < terms.size(); ++t) {
                        retVal.push_back({{i, j, s, t}});
                    }
                }
            }
        }
    }


    for (vector<vector<int>> comb : retVal) {
        vector<string> combStr;
        for (vector<int> subComb : comb) {
            string subCombStr = "";
            for (int singleTerm : subComb) {
                subCombStr += to_string(terms[singleTerm]) + "-";
            }
            subCombStr.pop_back();
            combStr.push_back(subCombStr);
        }
        retVal_string.push_back(combStr);
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

float getTopKFromMap (unordered_map<string, unordered_map<uint64_t, float>> &t_did_map, int k, int termConsidered, vector<uint32_t> &terms) {
    unordered_map<uint64_t, float> TopK;
    vector<pair<uint64_t, float>> TopKVec;

    vector<vector<string>> allPossibleSlicing = getAllPossibleSlicing(terms, termConsidered);

    for (vector<string> slicing : allPossibleSlicing) {
        unordered_map<uint64_t, float> DidScore;
        for (string comb : slicing) {
            if (t_did_map.find(comb) != t_did_map.end()) {
                for (auto item : t_did_map[comb]) {
                    DidScore[item.first] += item.second;
                }
            }
        }

        for (auto item : DidScore) {
            if (TopK.find(item.first) != TopK.end()) {
                TopK[item.first] = max(item.second, TopK[item.first]);
            } else {
                TopK[item.first] = item.second;
            }
        }
    }

    for (auto item : TopK) {
        TopKVec.push_back({item.first, item.second});
    }

    sort(TopKVec.begin(), TopKVec.end(), cmpDidTPair);
    if (TopKVec.size() < k) {
        return -2.0;
    }
    return TopKVec[k - 1].second;
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

    string cache_filename = argStr[0];
    int termConsidered = atoi(argStr[1].c_str());
    int d = k * atoi(argStr[2].c_str());

    unordered_set<string> cached_terms;

    load_exist_terms(cached_terms, "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms");

    if (termConsidered >= 2) {
        load_exist_terms(cached_terms, "/ssd2/home/bmmliu/logBaseFreq/2_term_freq_1.txt");
    }
    if (termConsidered >= 3) {
        load_exist_terms(cached_terms, "/ssd2/home/bmmliu/logBaseFreq/3_term_freq_1.txt");
    }
    if (termConsidered >= 4) {
        load_exist_terms(cached_terms, "/ssd2/home/bmmliu/logBaseFreq/4_term_freq_1.txt");
    }

    vector<float> realThreshold;

    vector<float> singleThreshold;
    vector<int> singleEstimatedK;
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
        float curEstimate;
        if (allTermResults.size() >= k) {
            allTermThreshold = allTermResults[k - 1].first;
        }

        realThreshold.push_back(allTermThreshold);

        // If doc less than k
        if (allTermThreshold == -1.0) {
            singleThreshold.push_back(-1.0);
            singleEstimatedK.push_back(-1);
            singleQueryTimeVec.push_back(-1.0);
            continue;
        }

        topk_queue topk(d);
        ranked_and_query wand_q(topk);

        vector<vector<string>> allPossibleSlicing = getAllPossibleSlicing(terms, termConsidered);
        unordered_map<string, unordered_map<uint64_t, float>> t_did_map;

        for (vector<string> slicing : allPossibleSlicing) {
            for (string comb : slicing) {
                if (cached_terms.find(comb) != cached_terms.end() && t_did_map.find(comb) == t_did_map.end()) {
                    vector<uint32_t> terms = getTermsFromString(comb);
                    Query subQuery;
                    subQuery.terms = terms;
                    auto cursor = make_max_scored_cursors(index, wdata, *scorer, subQuery);
                    wand_q(cursor, index.num_docs());
                    topk.finalize();
                    auto results = topk.topk();
                    topk.clear();

                    for(std::pair<float, uint64_t> result: results) {
                        t_did_map[comb][result.second] = result.first;
                    }
                }
            }
        }

        t_start = std::chrono::high_resolution_clock::now();
        curEstimate = getTopKFromMap(t_did_map, k, termConsidered, terms);
        t_end = std::chrono::high_resolution_clock::now();
        singleThreshold.push_back(curEstimate);
        singleQueryTimeVec.push_back(std::chrono::duration<double, std::milli>(t_end-t_start).count());

        if (curEstimate < 0) {
            singleEstimatedK.push_back(-2);
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
