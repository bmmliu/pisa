#include <iostream>
#include <optional>
#include <unordered_set>
#include <queue>
#include <random>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <sparsehash/dense_hash_set>
#include <sparsehash/dense_hash_map>
#include <bits/stdc++.h>

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

#define MaxQueryLen 16

using namespace pisa;
using term_id_type = uint32_t;
//using ext::hash;
using google::dense_hash_set;
using google::dense_hash_map;


struct Posting
{
    uint64_t did;
    short total_score;
    vector<short> scores;
    string term_string;
    int rank;

    Posting(uint64_t did, short total_score, vector<short> scores, string term_string, int rank)
        : did(did), total_score(total_score), scores(scores), term_string(term_string), rank(rank) {}
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


double nCr(double n, double r) {
    double sum = 1;

    for(int i = 1; i <= r; i++){
        sum = sum * (n - r + i) / i;
    }

    //return (int)sum;
    return std::exp(std::lgamma(n + 1)- std::lgamma(r + 1) - std::lgamma(n - r + 1));
}

double calculateO(int k, int kPrime, double s) {
    double res = 0.0;


    for (int i = kPrime; i < k; i++) {
        res += nCr(k - 1, i) * std::pow(s, i) * std::pow(1 - s, k - i - 1);
    }

    return res;
}

int getKPrime(int k, float s, float target_O) {

    for (int k_prime = 1; k_prime < k; k_prime++) {
        if (calculateO(k, k_prime, s) <= target_O) {
            return k_prime;
        }
    }

    return -1;
}



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

unordered_set<uint64_t> getSampleDid(string sample_file_path) {
    /*int sampleSize = didMax * samplePercent;
    std::vector<int> numbers(didMax+1);
    std::iota(numbers.begin(), numbers.end(), 0);
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(numbers.begin(), numbers.end(), g);
    std::unordered_set<uint64_t> result(numbers.begin(), numbers.begin() + sampleSize);*/

    std::unordered_set<uint64_t> result;
    string did_string;
    ifstream MyReadFile(sample_file_path);
    while (getline (MyReadFile, did_string)) {
        result.insert(stoull(did_string));
    }

    // Close the file
    MyReadFile.close();
    return result;
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

    int sample_percentage = atoi(argStr[0].c_str());

    float target_over_estimate_rate = atof(argStr[2].c_str());

    string sample_did_file_name = "/ssd2/home/bmmliu/SampleDid/" + to_string(sample_percentage) + "PercentSampleDid.txt";

    unordered_set<uint64_t> SampleDid = getSampleDid(sample_did_file_name);


    int KPrime = getKPrime(k, sample_percentage / 100.0, target_over_estimate_rate);
    cout << "k = " << k << endl;
    cout << "target O = " << target_over_estimate_rate << endl;
    cout << "k prime = " << KPrime << endl;

    vector<float> singleThreshold;

    int count = 0;
    for (auto const& query: queries) {
        auto terms = query.terms;


        topk_queue topk_old(k * 50);
        wand_query wand_q_old(topk_old);

        // calculate all terms threshold
        auto cursors = make_max_scored_cursors(index, wdata, *scorer, query);
        wand_q_old(cursors, index.num_docs());
        topk_old.finalize();
        auto allTermResults = topk_old.topk();
        topk_old.clear();
        float allTermThreshold = -1.0;

        int validDidCount = 0;
        for (auto post : allTermResults) {
            if (SampleDid.find(post.second) != SampleDid.end()) {
                validDidCount++;
            }
            if (validDidCount == KPrime) {
                allTermThreshold = post.first;
                break;
            }
        }

        singleThreshold.push_back(allTermThreshold);


        count++;
        if (count % 10 == 0) {
            clog << count << "queries sampling k = " << k << " target O = " << target_over_estimate_rate << " processed -- combine terms" << endl;
        }
    }


    std::cout << "single threshold" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < singleThreshold.size(); i++) {
        cout << singleThreshold[i] << '\n';
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
