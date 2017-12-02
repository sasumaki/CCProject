from collections import Counter

import fetchImages.catalog as catalog
from pprint import pprint

if __name__ == '__main__':
    catalog_ = catalog.load_catalog()
    year_ranges = Counter(catalog.time_frames(catalog_))
    schools = Counter(catalog.schools(catalog_))
    types = Counter(catalog.types(catalog_))
    forms = Counter(catalog.forms(catalog_))
    pprint(year_ranges)
    pprint(schools)
    pprint(types)
    pprint(forms)

    print(catalog.transform_categorical_to_numerical(catalog.time_frames(catalog_)))
