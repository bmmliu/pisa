#include <iostream>
#include <optional>
#include <unordered_set>

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
#include "scorer/scorer.hpp"

#include "CLI/CLI.hpp"

using namespace pisa;

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

vector<string> split (const string &s, char delim) {
    vector<string> result;
    stringstream ss (s);
    string item;

    while (getline (ss, item, delim)) {
        result.push_back (item);
    }

    return result;
}

string term_to_string (vector<uint32_t> &terms) {
    string sorted_terms_string = "";
    for (uint32_t term_id: terms) {
        sorted_terms_string += to_string(term_id) + "-";
    }
    sorted_terms_string.pop_back();

    return sorted_terms_string;
}

vector<Query> make_exist_term_queries(std::string exist_term_file) {
    std::vector<::pisa::Query> q;
    std::string m_term_lexicon = "/home/bmmliu/data/cw09b/CW09B.fwd.termlex";
    auto parse_query = resolve_query_parser(q, m_term_lexicon, std::nullopt, std::nullopt);
    std::ifstream is(exist_term_file);
    io::for_each_line(is, parse_query);
    return q;
}

void load_exist_terms(unordered_map<string, unordered_map<int, short>> &t_did_map, string gram_filename, string prefix_filename, string lexicon_filename, int num_of_term) {
    vector<Query> exist_terms_queries = make_exist_term_queries(gram_filename);
    ifstream in_lex(lexicon_filename);
    ifstream prefix_binary;
    prefix_binary.open(prefix_filename, std::ios::in | std::ios::binary);
    std::unordered_map<std::string, std::pair<int64_t, int64_t>> lex;
    int count = 0;
    for (auto const& query: exist_terms_queries) {
        // Get sorted query and convert it to string

        vector<uint32_t> sorted_terms = query.terms;
        sort(sorted_terms.begin(), sorted_terms.end());

        string sorted_terms_string = term_to_string(sorted_terms);

        string lexline;
        vector<std::string> lex_pair;

        std::getline(in_lex, lexline);
        boost::split(lex_pair, lexline, boost::is_any_of(" "), boost::token_compress_on);
        lex[sorted_terms_string] = pair<int64_t, int64_t>(std::stoll(lex_pair[0], nullptr, 0), std::stoll(lex_pair[1], nullptr, 0));

        int64_t start_pos = lex[sorted_terms_string].first;
        int64_t end_pos = lex[sorted_terms_string].second;
        int64_t cur_pos = start_pos + 4;
        while (cur_pos < end_pos) {
            prefix_binary.seekg(cur_pos);

            int did;
            short total_score = 0;

            prefix_binary.read(reinterpret_cast<char*>(&did), 4);
            for (int i = 0; i < num_of_term; i++) {
                unsigned char score;
                prefix_binary.read(reinterpret_cast<char*>(&score), 1);
                total_score += (short)score;
            }

            t_did_map[sorted_terms_string][did] = total_score;
            cur_pos = cur_pos + 4 + num_of_term;
        }
        count++;
        if (count % 100000 == 0) {
            clog << count << " " << num_of_term << " term combine loaded" << endl;
        }
        //cout << sorted_terms_string << "    " << did << "    " << total_score << endl;

    }

    in_lex.close();
    prefix_binary.close();
}

double getTopKFromMap(unordered_map<string, unordered_map<int, short>> &t_did_map, int k, int considered_term, vector<uint32_t> &terms) {
    vector<short> scores;
    sort(terms.begin(), terms.end());

    short threshold = 0;

    for (size_t i = 0; i < terms.size(); ++i) {
        vector<uint32_t> chosen_terms = {terms[i]};
        string cur_term_string = term_to_string(chosen_terms);
        if (t_did_map.find(cur_term_string) != t_did_map.end()) {
            //cout << "hit" << "     " << t_did_map[cur_term_string].size() << "     ";
            vector<short> TopK;
            for (auto did_score : t_did_map[cur_term_string]) {
                TopK.push_back(did_score.second);
            }

            sort(TopK.begin(), TopK.end(), greater<short>());
            //cout << TopK.size() << endl;
            if (TopK.size() >= k) {
                threshold = max(threshold, TopK[k - 1]);
                //cout << threshold << endl;
            }
        }
    }
    /*if (terms.size() >= 2 && considered_term >= 2) {
        for (size_t i = 0; i < terms.size(); ++i) {
            for (size_t j = i + 1; j < terms.size(); ++j) {
                vector<uint32_t> chosen_terms = {terms[i], terms[j]};
                string cur_term_string = term_to_string(chosen_terms);
                if (t_did_map.find(cur_term_string) != t_did_map.end()) {
                    vector<short> TopK;
                    for (auto did_score : t_did_map[cur_term_string]) {
                        TopK.push_back(did_score.second);
                    }

                    sort(TopK.begin(), TopK.end());

                    if (TopK.size() >= k) {
                        threshold = max(threshold, TopK[k - 1]);
                    }
                }
            }
        }
    }

    if (terms.size() >= 3 && considered_term >= 3) {
        for (size_t i = 0; i < terms.size(); ++i) {
            for (size_t j = i + 1; j < terms.size(); ++j) {
                for (size_t s = j + 1; s < terms.size(); ++s) {
                    vector<uint32_t> chosen_terms = {terms[i], terms[j], terms[s]};
                    string cur_term_string = term_to_string(chosen_terms);
                    if (t_did_map.find(cur_term_string) != t_did_map.end()) {
                        vector<short> TopK;
                        for (auto did_score : t_did_map[cur_term_string]) {
                            TopK.push_back(did_score.second);
                        }

                        sort(TopK.begin(), TopK.end());

                        if (TopK.size() >= k) {
                            threshold = max(threshold, TopK[k - 1]);
                        }
                    }
                }
            }
        }
    }

    if (terms.size() >= 4 && considered_term >= 4) {
        for (size_t i = 0; i < terms.size(); ++i) {
            for (size_t j = i + 1; j < terms.size(); ++j) {
                for (size_t s = j + 1; s < terms.size(); ++s) {
                    for (size_t t = s + 1; t < terms.size(); ++t) {
                        vector<uint32_t> chosen_terms = {terms[i], terms[j], terms[s], terms[t]};
                        string cur_term_string = term_to_string(chosen_terms);
                        if (t_did_map.find(cur_term_string) != t_did_map.end()) {
                            vector<short> TopK;
                            for (auto did_score : t_did_map[cur_term_string]) {
                                TopK.push_back(did_score.second);
                            }

                            sort(TopK.begin(), TopK.end());

                            if (TopK.size() >= k) {
                                threshold = max(threshold, TopK[k - 1]);
                            }
                        }
                    }
                }
            }
        }
    }*/

    return (double)threshold;
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


    int getDandTFromFileNameFlag = 1;
    vector<string> argStr;

    if (exist_term_filename && getDandTFromFileNameFlag) {
        argStr = split(*exist_term_filename, ',');
    }

    string cache_filename = argStr[0];
    int considered_term = atoi(argStr[1].c_str());


    vector<float> singleThreshold;
    vector<float> singleQueryTimeVec;

    auto t_start = std::chrono::high_resolution_clock::now();
    auto t_end = std::chrono::high_resolution_clock::now();

    unordered_map<string, unordered_map<int, short>> t_did_map;

    string single_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Lexicon/CW09B.fwd.terms";
    string single_prefix_path = "/hdd1/jg6226/data/Hit_Ratio_Project/Threshold_Estimation/First_Layer_Index/single_with_termscore/single_prefix";
    string single_lexicon_path = "/hdd1/jg6226/data/Hit_Ratio_Project/Threshold_Estimation/First_Layer_Index/single_with_termscore/single_lexicon.txt";

    string duplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/duplet_cleaned.txt";
    string duplet_prefix_path = "/hdd1/jg6226/data/Hit_Ratio_Project/Threshold_Estimation/First_Layer_Index/duplet_with_termscore/duplet_prefix";
    string duplet_lexicon_path = "/hdd1/jg6226/data/Hit_Ratio_Project/Threshold_Estimation/First_Layer_Index/duplet_with_termscore/duplet_lexicon.txt";

    string triplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/triplet_cleaned.txt";
    string triplet_prefix_path = "/hdd1/jg6226/data/Hit_Ratio_Project/Threshold_Estimation/First_Layer_Index/triplet_with_termscore/triplet_prefix";
    string triplet_lexicon_path = "/hdd1/jg6226/data/Hit_Ratio_Project/Threshold_Estimation/First_Layer_Index/triplet_with_termscore/triplet_lexicon.txt";

    string quadruplet_gram_path = "/home/jg6226/data/Hit_Ratio_Project/Real_Time_Query_System/Prefix_Grams/quadruplet_cleaned.txt";
    string quadruplet_prefix_path = "/hdd1/jg6226/data/Hit_Ratio_Project/Threshold_Estimation/First_Layer_Index/triplet_with_termscore/quadruplet_prefix";
    string quadruplet_lexicon_path = "/hdd1/jg6226/data/Hit_Ratio_Project/Threshold_Estimation/First_Layer_Index/triplet_with_termscore/quadruplet_lexicon.txt";

    load_exist_terms(t_did_map, single_gram_path, single_prefix_path, single_lexicon_path, 1);
    if (considered_term >= 2) {
        load_exist_terms(t_did_map, duplet_gram_path, duplet_prefix_path, duplet_lexicon_path, 2);
    }
    if (considered_term >= 3) {
        load_exist_terms(t_did_map, triplet_gram_path, triplet_prefix_path, triplet_lexicon_path, 3);
    }
    if (considered_term >= 4) {
        load_exist_terms(t_did_map, quadruplet_gram_path, quadruplet_prefix_path, quadruplet_lexicon_path, 4);
    }

    for (auto const& query: queries) {

        if(ifDupTerm(query.terms)) {
            singleThreshold.push_back(-1.0);
            singleQueryTimeVec.push_back(-1.0);
            continue;
        }

        auto terms = query.terms;

        float threshold = 0;

        t_start = std::chrono::high_resolution_clock::now();
        threshold = getTopKFromMap(t_did_map, k, considered_term, terms);
        t_end = std::chrono::high_resolution_clock::now();

        singleThreshold.push_back(threshold);

        singleQueryTimeVec.push_back(std::chrono::duration<double, std::milli>(t_end-t_start).count());

    }


    std::cout << "single threshold" << endl;
    std::cout << "*************************************************************" << endl;
    for(int i = 0; i < singleThreshold.size(); i++) {
        cout << singleThreshold[i] << '\n';
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
