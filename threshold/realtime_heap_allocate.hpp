#pragma once

#include <queue>
#include <numeric>
#include <filesystem>
#include <bitset>
#include <unordered_map>
#include <utility>
#include <vector>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <utility>

#include <boost/algorithm/string/join.hpp>
#include <boost/algorithm/string.hpp>
#include "cppitertools/combinations.hpp"

#include "query/algorithm/extern_query.hpp"
#include "index_types.hpp"
#include "cursor/max_scored_cursor.hpp"
#include "wand_data_raw.hpp"
#include "wand_data.hpp"
#include "wand_data_compressed.hpp"
#include "query/algorithm.hpp"
#include "pisa/tools/app.hpp"
#include "first_layer_index.hpp"

using namespace pisa;

namespace candidates {
using lexicon_offset = std::pair<int64_t, int64_t>;
using lexicon = std::unordered_map<std::string, lexicon_offset>;
using index_path = std::tuple<std::string, std::string, std::string>;
using comb_grams = std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>, std::vector<std::string>>;

using wand_raw_index = wand_data<wand_data_raw>;
using wand_uniform_index = wand_data<wand_data_compressed<>>;
using wand_uniform_index_quantized = wand_data<wand_data_compressed<PayloadType::Quantized>>;
using query_function = std::function<std::vector<typename topk_queue::entry_type>(Query)>;
using scorers = std::unique_ptr<index_scorer<wand_data<wand_data_raw>>, std::default_delete<index_scorer<wand_data<wand_data_raw>>>>;
using documents = Payload_Vector<std::string_view>;
using sources = std::shared_ptr<mio::mmap_source>;
using terms = Payload_Vector<std::string_view>;

struct struct_path
{
    index_path sinind = {};
    index_path dupind = {};
    index_path triind = {};
    index_path quadind = {};
};

struct struct_lexicon
{
    lexicon sinsublex = {};
    lexicon dupsublex = {};
    lexicon trisublex = {};
    lexicon quadsublex = {};
};

struct struct_input
{
    std::ifstream sinsub;
    std::ifstream dupsub;
    std::ifstream trisub;
    std::ifstream quadsub;
};

struct query_res {
    const Query term;
    std::vector<uint64_t> dids;
};

class realtime_heap_allocate {
  public:
    lexicon load_lexicon(const std::string& gram_path, std::string lexicon_path);
    inline lexicon_offset get_offset(const std::string& substructure, lexicon & substructure_lex);
    // std::unordered_map<int, std::vector<float>> compute_hit_ratio(std::string query, std::vector<int> & budget, std::vector<int> & topk, std::bitset<4> & types, std::string output_path, const std::string & metrics);
    std::unordered_map<int, std::vector<float>> compute_hit_ratio(std::string query, std::vector<int> & budget, const std::vector<int> & topk, std::bitset<4> & types, const std::string & metric);
    void get_accuracy(std::string set_path, std::vector<int> & budget, std::vector<int> & topk, std::bitset<4> types, const std::string & metric = "Recall");
    comb_grams get_comb(std::vector<std::string> & terms, std::bitset<4> & comb_type);
    std::vector<std::string> split(const std::string& str, const std::string& delim);
    float metric_score(std::string metric, const std::unordered_map<std::string, bool> & topk_dict, std::vector<std::string> & topk_lst, std::vector<std::string> & topk_pred);
    void initialize();
    realtime_heap_allocate(index_path single, index_path duplet, index_path triplet, index_path quadruplet);
    std::vector<postings> combine_postings(std::vector<postings> & postingslist);
    std::tuple<std::vector<std::string>, int, int, std::vector<short>> lookup_allocation(std::vector<postings> & posting_from, int budget, int k);
    template <typename IndexType, typename WandType>
    double did_score(IndexType & index, WandType & wdata, scorers & scorer, query_function & query_fun, documents & docmap,
                     App<arg::Index, arg::WandData<arg::WandMode::Required>, arg::Query<arg::QueryMode::Ranked>, arg::Algorithm, arg::Scorer, arg::Thresholds, arg::Threads> & app, std::string term, uint64_t did)
    {
        termdidlist_search td_s;
        pisa::Query q = app.lookup(std::move(term));
        std::vector<uint64_t> dids{did};
        std::vector<double> lookup_score = td_s(make_max_scored_cursors(index, wdata, *scorer, q), dids);
        return lookup_score[0];
    };
    template <typename IndexType, typename WandType>
    std::vector<double> did_score_list(IndexType & index, WandType & wdata, scorers & scorer, query_function & query_fun, documents docmap,
                                       sources source,
                                       App<arg::Index, arg::WandData<arg::WandMode::Required>,
                                           arg::Query<arg::QueryMode::Ranked>, arg::Algorithm, arg::Scorer, arg::Thresholds, arg::Threads> & app,
                                       std::string term, std::vector<uint64_t> & vecdid)
    {
        termdidlist_search td_s;
        pisa::Query q = app.lookup(std::move(term));
        std::vector<double> lookup_score = td_s(make_max_scored_cursors(index, wdata, *scorer, q), vecdid);
        return lookup_score;
    };

    template <typename IndexType, typename WandType>
    std::vector<double> did_score_list_modified(IndexType & index, WandType & wdata, scorers & scorer, query_function & query_fun, const TermProcessor & term_processor,
                                                std::string & term, std::vector<uint64_t> & vecdid, std::unique_ptr<pisa::Tokenizer> & tokenizer)
    {
        termdidlist_search td_s;
        std::vector<double> lookup_score = td_s(make_max_scored_cursors(index, wdata, *scorer, pisa::parse_query_terms(term, *tokenizer, term_processor)), vecdid);
        return lookup_score;
    };

    template <typename IndexType, typename WandType>
    std::vector<double> did_score_list_final(IndexType & index, WandType & wdata, std::unique_ptr<index_scorer<WandType>, std::default_delete<index_scorer<WandType>>> & scorer, Query single_term_query, std::vector<uint64_t> & vecdid)
    {
        termdidlist_search td_s;
        auto cursors = make_max_scored_cursors(index, wdata, *scorer, single_term_query);
        std::vector<double> lookup_score = td_s(cursors, vecdid);
        return lookup_score;
    };

    ~realtime_heap_allocate();
  private:
    inline std::vector<short> combine_twolists(std::vector<short> & list1, std::vector<short> & list2);
    inline postings combine_substructures(std::vector<postings> & substructures);

  private:
    std::vector<int> budgets = {};
    std::vector<std::string> query_name = {};
    struct_path structPath = {};
    struct_lexicon structLexicon = {};
    struct_input structInput = {};
};
}

